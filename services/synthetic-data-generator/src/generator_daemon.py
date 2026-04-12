import time
import logging
import random
import datetime
import pandas as pd
import numpy as np
import s3fs
import json
from config_schema import StateStore, SimulationConfig, ServiceConfig

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

class DataGeneratorDaemon:
    def __init__(self):
        self.state_store = StateStore(DB_PATH)
        logger.info("Loading baseline SECOM dataset into memory...")
        self.baseline_df = pd.read_csv(RAW_DATA_PATH)

        # Ensure 'Time' is parsed as datetime objects
        self.baseline_df['Time'] = pd.to_datetime(self.baseline_df['Time'])
        
        # Create a helper column for grouping by calendar day
        self.baseline_df['Date_Block'] = self.baseline_df['Time'].dt.date
        
        # Get a sorted list of unique days in the dataset to iterate through
        self.unique_dates = sorted(self.baseline_df['Date_Block'].unique())
        
        # Initialize pointer to track sequential progress
        self.current_date_idx = 0

        self.numeric_cols = self.baseline_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude targets or timestamp identifiers from noise/drift
        self.features_to_mutate = [c for c in self.numeric_cols if c not in ['Time', 'Target', 'Pass_Fail']]
        self.feature_stds = self.baseline_df[self.features_to_mutate].std()
        
        # S3 Filesystem setup for MinIO
        self.fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': S3_ENDPOINT},
            key=S3_ACCESS_KEY, 
            secret=S3_SECRET_KEY
        )

    def _get_next_day_block(self) -> pd.DataFrame:
        """Samples all sequential rows for a specific day to preserve time-series behavior."""
        if not self.unique_dates:
            return self.baseline_df.copy() # Fallback if dates couldn't be parsed
        
        # Identify the date for this cycle
        target_date = self.unique_dates[self.current_date_idx]
        
        # Filter the dataframe for the targeted day
        day_block_df = self.baseline_df[self.baseline_df['Date_Block'] == target_date].copy()
        
        # Advance the index, loop back to 0 if we hit the end of the dataset
        self.current_date_idx = (self.current_date_idx + 1) % len(self.unique_dates)
        
        # Drop the helper column before returning so it doesn't leak into downstream storage
        return day_block_df.drop(columns=['Date_Block'])

    def apply_mutations(self, df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
        """Applies controlled jitter and targeted sigma shifts."""
        # 1. Controlled Jitter (Micro-noise)
        if config.jitter_variance > 0:
            noise = np.random.normal(
                loc=0, 
                scale=self.feature_stds * config.jitter_variance, 
                size=(len(df), len(self.features_to_mutate))
            )
            # Only apply noise where data is not null to preserve missingness topology
            mask = df[self.features_to_mutate].notna()
            df[self.features_to_mutate] = df[self.features_to_mutate].where(
                ~mask, df[self.features_to_mutate] + noise
            )

        # 2. Targeted Drift (Sigma Shift)
        for feature, sigma in config.drift_config.items():
            if feature in self.features_to_mutate:
                shift_value = self.feature_stds[feature] * sigma
                df[feature] = df[feature] + shift_value

        return df

    def run_cycle(self):
        config = self.state_store.get_config()
        
        if not config.is_running:
            logger.info("Daemon sleeping. State: is_running=False")
            time.sleep(5)
            return

        logger.info(f"Generating batch of {config.batch_size} wafers...")
        
        # 1. Generate Data
        batch_df = self._get_next_day_block()
        mutated_df = self.apply_mutations(batch_df, config)

        if len(batch_df) > config.batch_size:
            # Sample N rows and sort chronologically
            mutated_df = mutated_df.sample(n=config.batch_size).sort_values('Time')

        # 2. Inject Provenance Metadata
        now = datetime.datetime.now(datetime.timezone.utc)
        mutated_df['is_synthetic'] = True
        mutated_df['generation_timestamp'] = now
        mutated_df['applied_drift_features'] = json.dumps(config.drift_config)

        # 3. Write to MinIO (Hive Partitioned)
        partition_path = f"{S3_BUCKET}/year={now.year}/month={now.month:02d}/day={now.day:02d}"
        file_path = f"{partition_path}/batch_{int(now.timestamp())}.parquet"
        
        try:
            mutated_df.to_parquet(file_path, filesystem=self.fs, index=False)
            logger.info(f"Successfully wrote batch to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write to MinIO: {e}")

        # Sleep for the configured interval
        time.sleep(config.generation_interval_seconds)

if __name__ == "__main__":
    daemon = DataGeneratorDaemon()
    logger.info("Generator Daemon Started.")
    while True:
        try:
            daemon.run_cycle()
        except Exception as e:
            logger.error(f"Fatal error in daemon cycle: {e}")
            time.sleep(5) # Prevent tight crash loop