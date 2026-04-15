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

SHIFT = [
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
        self.baseline_df = pd.read_csv(RAW_DATA_PATH)
        self.baseline_df['Time'] = pd.to_datetime(self.baseline_df['Time'])
        self.baseline_df['Date_Block'] = self.baseline_df['Time'].dt.date
        self.unique_dates = sorted(self.baseline_df['Date_Block'].unique())
        
        # Per-line date pointer so each line advances independently
        self._date_ptrs: dict[str, int] = {}
        self._lot_counters: dict[str, int] = {}

        numeric_cols = self.baseline_df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude targets or timestamp identifiers from noise/drift
        self.features_to_mutate = [c for c in self.numeric_cols if c not in ['Time', 'Target', 'Pass_Fail']]
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
                await self.js.add_stream(name=NATS_STREAM, subjects=[NATS_SUBJECT])
                logger.info(f"JetStream 'SECOM_PIPELINE' initialized for subject '{NATS_SUBJECT}'")
            except Exception as e:
                logger.debug("Stream already exists: %s", e)
            logger.info("Connected to NATS at %s", NATS_URL)

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            self.nc = None
            self.js = None

    def _next_day_block(self, line_id:str) -> pd.DataFrame:
        """Samples all sequential rows for a specific day to preserve time-series behavior."""
        ptr = self._date_ptrs.get(line_id, 0)
        target_date = self.unique_dates[ptr % len(self.unique_dates)] # Safeguarding the Target Date, belt-and-suspenders safety net
        self._date_ptrs[line_id] = (ptr+1) % len(self.unique_dates) # Advance the index, loop back to 0 if we hit the end of the dataset

        if not self.unique_dates:
            return self.baseline_df.copy() # Fallback if dates couldn't be parsed
        
        return self.baseline_df[self.baseline_df["Date_Block"] == target_date].copy().drop(columns=["Date_Block"])

    def _apply_mutations(self, df: pd.DataFrame, lc: LineConfig) -> pd.DataFrame:
        """Applies controlled jitter and targeted sigma shifts."""
        # 1. Controlled Jitter (Micro-noise)
        if lc.jitter_variance > 0:
            noise = np.random.normal(
                loc=0, 
                scale=self.feature_stds * lc.jitter_variance, 
                size=(len(df), len(self.features_to_mutate))
            )
            # Only apply noise where data is not null to preserve missingness topology
            mask = df[self.features_to_mutate].notna()
            df[self.features_to_mutate] = df[self.features_to_mutate].where(
                ~mask, df[self.features_to_mutate] + noise
            )

        # 2. Targeted Drift (Sigma Shift)
        for feature, sigma in lc.drift_config.items():
            if feature in self.features_to_mutate:
                df[feature] = df[feature] + self.feature_stds[feature] * sigma

        return df

    def _next_lot_id(self, line_id:str) -> str:
        n = self._lot_counters.get(line_id, 0)
        self._lot_counters[line_id] = n + 1 
        return f"{line_id}-LOT-{n:05d}"

    async def run_line_cycle(self, line_id: str, lc: LineConfig) -> None:
        now = datetime.datetime.now(datetime.timezone.utc)

        if lc.fault_injection_enabled and random.random() < lc.fault_probability:
            logger.warning("[%s] Fault injected — pausing for %ds", line_id, lc.fault_duration_seconds)
            self.state_store.log_event(line_id, "FAULT", {"duration_s": lc.fault_duration_seconds})
            await asyncio.sleep(lc.fault_duration_seconds)

            return 

        batch_df = self._next_day_block(line_id)
        mutated_df = self._apply_mutations(batch_df, lc)

        if len(batch_df) > lc.batch_size:
            # Sample N rows and sort chronologically
            mutated_df = mutated_df.sample(n=config.batch_size).sort_values('Time')
        
        lot_id = self._next_lot_id
        shift = _current_shift(mutated_df['Time'].hour)
        iso_ts = now.isoformat()

        mutated_df = mutated_df.copy()

        mutated_df["line_id"]              = line_id
        mutated_df["tester_id"]            = lc.tester_id
        mutated_df["shift"]                = shift
        mutated_df["lot_id"]               = lot_id
        mutated_df["is_synthetic"]         = True
        mutated_df["generation_timestamp"] = now
        mutated_df["applied_drift_features"] = json.dumps(lc.drift_config)

        # 3. Write to MinIO (Hive Partitioned)
        partition = (
            f"{S3_BUCKET}/line_id={line_id}"
            f"/year={now.year}/month={now.month:02d}/day={now.day:02d}"
        )

        file_path = f"{partition_path}/batch_{int(now.timestamp())}.parquet"

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
    
    async def publish_metadata(self, payload: dict) -> None:
        """Publishes the generation event payload to NATS."""
        try:
            # NATS requires bytes
            message = json.dumps(payload).encode('utf-8')
            ack = await self.js.publish(NATS_SUBJECT, message)
            
            logger.info("[%s] NATS ack seq=%d", line_id, ack.seq)
        except Exception as e:
            logger.error("[%s] NATS publish failed: %s", line_id, e)

    async def start(self):
        """Main async entrypoint."""
        await self._connect_nats()
        logger.info("Generator Daemon Started.")

        try:
            while True:
                config: SimulationConfig = self.state_store.get_config()

                task = []
                for line_id, lc in config.lines.items():
                    if lc.is_running:
                        tasks.append(self.run_line_cycle(line_id, lc))
                    else:
                        logger.debug("[%s] idle", line_id)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                min_interval = min(
                    (lc.generation_interval_seconds for lc in config.lines.values() if lc.is_running),
                    default=5,
                )
                await asyncio.sleep(min_interval)
                
        finally:
            if self.nc and not self.nc.is_closed:
                await self.nc.drain()
                logger.info("NATS connection drained and closed.")

if __name__ == "__main__":
    daemon = DataGeneratorDaemon()
    asyncio.run(daemon.start())