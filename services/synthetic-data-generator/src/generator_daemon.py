"""
generator_daemon.py — Multi-line SECOM synthetic data generator.
 
Runs LINE_A, LINE_B, and LINE_C as independent asyncio tasks, each with
its own is_running state, drift config, fault injection, and MinIO path.
Events are published to NATS JetStream with line_id metadata so Airflow
can route them to the correct bronze partitions.
"""

import time
import logging
import random
import datetime
import pandas as pd
import numpy as np
import s3fs
import json
import nats
import asyncio
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError
from nats.js.api import StorageType, RetentionPolicy, DiscardPolicy
from config_schema import LineConfig, StateStore, SimulationConfig, ServiceConfig

# Configure Production Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

service_config = ServiceConfig()

# Constants
S3_BUCKET = service_config.minio_bucket
S3_ENDPOINT = service_config.minio_endpoint 
S3_ACCESS_KEY = service_config.minio_access_key
S3_SECRET_KEY = service_config.minio_secret_key
RAW_DATA_PATH = service_config.raw_dataset_file
DB_PATH = service_config.db_path

NATS_ENDPOINT = service_config.nats_endpoint
NATS_SUBJECT = service_config.nats_subject
NATS_STREAM = service_config.nats_stream_name

SHIFTS = [
    ("Day", 6, 14),
    ("Swing", 14, 22),
    ("Night", 22, 30), # 30 wraps to 06
]

def _current_shift(hour: int) -> str:
    for name, start, end in SHIFTS:
        if start <= hour < end or (end > 24 and (hour >= start or hour < end - 24)):
            return name
    return "Night"

class DataGeneratorDaemon:
    def __init__(self):
        self.state_store = StateStore(DB_PATH)

        logger.info("Loading baseline SECOM dataset into memory...")
        self.baseline_df = pd.read_csv(RAW_DATA_PATH, parse_dates=['Time'], date_format='%Y-%m-%d %H:%M:%S')

        self.baseline_df = self.baseline_df.copy()
        self.baseline_df['Date_Block'] = self.baseline_df['Time'].dt.date
        self.unique_dates = sorted(self.baseline_df['Date_Block'].unique())
        
        # Dictionary to track fault cooldowns independently and next run time per line
        self.fault_cooldowns: dict[str, datetime.datetime] = {}
        self.next_run_times: dict[str, datetime.datetime] = {}

        numeric_cols = self.baseline_df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude targets or timestamp identifiers from noise/drift
        self.features_to_mutate = [c for c in numeric_cols if c not in ['Time', 'Target', 'Pass_Fail']]
        self.feature_stds = self.baseline_df[self.features_to_mutate].std()
        
        # S3 Filesystem setup for MinIO
        self.fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': S3_ENDPOINT},
            key=S3_ACCESS_KEY, 
            secret=S3_SECRET_KEY
        )

        self.nc = None
        self.js = None

    async def _connect_nats(self):
        """Establish connection to the NATS broker."""
        try:
            self.nc = await nats.connect(NATS_ENDPOINT)
            self.js = self.nc.jetstream()

            # Ensure the stream exists (Idempotent operation)
            try:
                await self.js.add_stream(
                    name=NATS_STREAM, 
                    subjects=[NATS_SUBJECT],
                    storage=StorageType.FILE,
                    retention=RetentionPolicy.WorkQueue,
                    discard=DiscardPolicy.OLD,
                    )
                logger.info(f"JetStream 'SECOM_PIPELINE' initialized for subject '{NATS_SUBJECT}'")
            except Exception as e:
                logger.debug("Stream already exists: %s", e)
            logger.info("Connected to NATS at %s", NATS_ENDPOINT)

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            self.nc = None
            self.js = None

    def _next_day_block(self, lc:LineConfig) -> pd.DataFrame:
        """Samples all sequential rows for a specific day to preserve time-series behavior."""
        if not self.unique_dates:
            return self.baseline_df.copy() # Fallback if dates couldn't be parsed

        target_date = self.unique_dates[lc.date_ptr % len(self.unique_dates)] # Safeguarding the Target Date, belt-and-suspenders safety net
        lc.date_ptr += 1

        if lc.date_ptr >= len(self.unique_dates):
            lc.date_ptr = 0
            lc.year_offset += 1
        
        df = self.baseline_df[self.baseline_df["Date_Block"] == target_date].copy().drop(columns=["Date_Block"])
        
        if lc.year_offset > 0:
            df['Time'] = df['Time'] + pd.DateOffset(years=lc.year_offset)
        
        
        return df

    def _apply_mutations(self, df: pd.DataFrame, lc: LineConfig) -> pd.DataFrame:
        """Applies controlled jitter and targeted sigma shifts."""
        # 1. Controlled Jitter (Micro-noise) - LOCAL standard deviation for this specific day
        if lc.jitter_variance > 0:  
            local_stds = df[self.features_to_mutate].std().fillna(0)

            noise = np.random.normal(
                loc=0, 
                scale=local_stds * lc.jitter_variance, 
                size=(len(df), len(self.features_to_mutate))
            )
            # Only apply noise where data is not null to preserve missingness topology
            mask = df[self.features_to_mutate].notna()
            df[self.features_to_mutate] = df[self.features_to_mutate].where(
                ~mask, df[self.features_to_mutate] + noise
            )

        # 2. Targeted Drift (Sigma Shift) - GLOBAL feature_stds, equipment failure/drift should be an absolute physical shift, not a local one.
        for feature, sigma in lc.drift_config.items():
            if feature in self.features_to_mutate:
                df[feature] = df[feature] + self.feature_stds[feature] * sigma

        return df

    def _next_lot_id(self, lc:LineConfig) -> str:
        lc.lot_counter += 1
        return f"{lc.line_id}-LOT-{lc.lot_counter:05d}"

    async def run_line_cycle(self, line_id: str, lc: LineConfig) -> None:
        now = datetime.datetime.now(datetime.timezone.utc)

        next_run = self.next_run_times.get(line_id)
        if next_run and now < next_run:
            return

        cooldown_end = self.fault_cooldowns.get(line_id)
        if cooldown_end and now < cooldown_end:
            logger.debug("[%s] down due to fault (wakes at %s)", line_id, cooldown_end.strftime('%H:%M:%S'))
            return

        if lc.fault_injection_enabled and random.random() < lc.fault_probability:
            logger.warning("[%s] Fault injected — pausing for %ds", line_id, lc.fault_duration_seconds)
            self.state_store.log_event(line_id, "FAULT", {"duration_s": lc.fault_duration_seconds})
            self.fault_cooldowns[line_id] = now + datetime.timedelta(seconds=lc.fault_duration_seconds)
            return 

        batch_df = self._next_day_block(lc)
        mutated_df = self._apply_mutations(batch_df, lc)

        if len(batch_df) > lc.batch_size:
            # Sample N rows and sort chronologically
            mutated_df = mutated_df.sample(n=lc.batch_size).sort_values('Time')
        
        lot_id = self._next_lot_id(lc)
        shift = _current_shift(mutated_df['Time'].dt.hour.iloc[0])
        iso_ts = now.isoformat()

        mutated_df = mutated_df.copy()

        mutated_df["line_id"] = line_id
        mutated_df["tester_id"] = lc.tester_id
        mutated_df["shift"] = shift
        mutated_df["lot_id"] = lot_id
        mutated_df["is_synthetic"] = True
        mutated_df["generation_timestamp"] = now
        mutated_df["applied_drift_features"] = json.dumps(lc.drift_config)

        # 3. Write to MinIO (Hive Partitioned)
        partition = (
            f"{S3_BUCKET}/line_id={line_id}"
            f"/year={now.year}/month={now.month:02d}/day={now.day:02d}"
        )

        file_path = f"{partition}/batch_{int(now.timestamp())}.parquet"

        try:
            await asyncio.to_thread(mutated_df.to_parquet, file_path, filesystem=self.fs, index=False)
            logger.info("[%s] Wrote batch (%d wafers) → %s", line_id, len(mutated_df), file_path)
        except Exception as e:
            logger.error("[%s] MinIO write failed: %s", line_id, e)
            return

        if self.js:
            payload = {
                "event_type": "TEST_COMPLETED",
                "file_path": f"s3://{file_path}",
                "batch_size": len(mutated_df),
                "line_id": line_id,
                "tester_id": lc.tester_id,
                "shift": shift,
                "lot_id": lot_id,
                "is_synthetic": True,
                "generation_timestamp": iso_ts,
                "applied_drift_features": lc.drift_config,
            }
   
        await self.publish_metadata(payload)  
        self.state_store.log_event(line_id, "BATCH", {"lot_id": lot_id, "wafers": len(mutated_df)})
        self.next_run_times[line_id] = now + datetime.timedelta(seconds=lc.generation_interval_seconds)
    
    async def publish_metadata(self, payload: dict) -> None:
        """Publishes the generation event payload to NATS."""
        try:
            # NATS requires bytes
            message = json.dumps(payload).encode('utf-8')
            ack = await self.js.publish(NATS_SUBJECT, message)
            
            logger.info("[%s] NATS ack seq=%d", payload.get('line_id',"UNKNOWN"), ack.seq)
        except Exception as e:
            logger.error("[%s] NATS publish failed: %s", payload.get('line_id',"UNKNOWN"), e)

    async def start(self):
        """Main async entrypoint."""
        await self._connect_nats()
        logger.info("Generator Daemon Started.")

        try:
            while True:
                config: SimulationConfig = self.state_store.get_config()

                tasks = []
                for line_id, lc in config.lines.items():
                    if lc.is_running:
                        tasks.append(self.run_line_cycle(line_id, lc))
                    else:
                        logger.debug("[%s] idle", line_id)
                
                if tasks:
                    await asyncio.gather(
                        *tasks, 
                        # return_exceptions=True,   # Comment for Fail Fast
                        )
                
                self.state_store.update_config(config)
                await asyncio.sleep(1)
                
        finally:
            if self.nc and not self.nc.is_closed:
                await self.nc.drain()
                logger.info("NATS connection drained and closed.")

if __name__ == "__main__":
    daemon = DataGeneratorDaemon()
    asyncio.run(daemon.start())