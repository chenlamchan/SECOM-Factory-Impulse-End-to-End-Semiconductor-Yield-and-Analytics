import sqlite3
import json
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Optional
from pathlib import Path

class ServiceConfig(BaseSettings):
    minio_endpoint:str = Field(default="http://minio:9000")
    minio_bucket:str = Field(default="s3://data-lake")
    minio_access_key_file:str
    minio_secret_key_file:str
    db_path:str
    raw_dataset_file:str

    nats_endpoint:str = Field(default="nats://nats:4222")
    nats_subject:str = Field(default="secom.data.generated")

    @property
    def minio_access_key(self):
        return Path(self.minio_access_key_file).read_text().strip()

    @property
    def minio_secret_key(self):
        return Path(self.minio_secret_key_file).read_text().strip()

    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=False
    )

class SimulationConfig(BaseModel):
    """Pydantic model defining the strict schema for our simulation state."""
    is_running: bool = Field(default=False, description="Master switch for the daemon")
    batch_size: int = Field(default=25, ge=1, description="Simulated FOUP batch size")
    generation_interval_seconds: int = Field(default=10, ge=1, description="How often (seconds) to generate a batch")
    jitter_variance: float = Field(default=0.01, ge=0.0, description="Multiplier for micro-noise (fraction of std dev)")
    drift_config: Dict[str, float] = Field(default_factory=dict, description="Dictionary of Feature -> Sigma Shift")

class StateStore:
    """Handles thread-safe SQLite operations for the simulation state."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state_json TEXT NOT NULL
                )
            """)
            # Insert default state if empty
            cursor = conn.execute("SELECT COUNT(*) FROM config")
            if cursor.fetchone()[0] == 0:
                default_state = SimulationConfig().model_dump_json()
                conn.execute("INSERT INTO config (id, state_json) VALUES (1, ?)", (default_state,))

    def get_config(self) -> SimulationConfig:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT state_json FROM config WHERE id = 1")
            state_json = cursor.fetchone()[0]
            return SimulationConfig.model_validate_json(state_json)

    def update_config(self, config: SimulationConfig) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE config SET state_json = ? WHERE id = 1",
                (config.model_dump_json(),)
            )