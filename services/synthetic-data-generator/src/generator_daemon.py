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

    def _block_bootstrap(self, batch_size: int) -> pd.DataFrame:
        """Samples sequential rows to preserve time-series autocorrelation."""
        max_start_idx = len(self.baseline_df) - batch_size
        if max_start_idx < 0:
            return self.baseline_df.copy() # Fallback if batch > dataset
        
        start_idx = random.randint(0, max_start_idx)
        return self.baseline_df.iloc[start_idx : start_idx + batch_size].copy()

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
        batch_df = self._block_bootstrap(config.batch_size)
        mutated_df = self.apply_mutations(batch_df, config)

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