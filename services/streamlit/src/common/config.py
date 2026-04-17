"""
config_schema.py — Multi-line simulation state management.
Extends config to support LINE_A / LINE_B / LINE_C
with independent run state, drift configs, and tester assignments.
Backward-compatible: falls back to defaults on first boot.
"""

import os
import sqlite3
import json
from datetime import datetime
from dataclasses import asdict, dataclass
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
    nats_stream_name:str = Field(default="SECOM_PIPELINE")

    trino_host:str = Field(default="trino")
    trino_port:str = Field(default="8080")
    catalog_name:str = "secom_catalog"

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

class LineConfig(BaseSettings):
    """Per-production-line simulation parameters."""
    line_id: str
    tester_id: str = ""
    is_running: bool = Field(default=False, description="Master switch for the daemon")
    batch_size: int = Field(default=25, ge=1, description="Simulated FOUP batch size")
    generation_interval_seconds: int = Field(default=30, ge=1, description="How often (seconds) to generate a batch")
    jitter_variance: float = Field(default=0.01, ge=0.0, description="Multiplier for micro-noise (fraction of std dev)")
    drift_config: Dict[str, float] = Field(default_factory=dict, description="Dictionary of Feature -> Sigma Shift")

    fault_injection_enabled: bool = Field(default=False, description="To enable the fault happens for OEE simulation")
    fault_probability: float = Field(default=0.05,ge=0) # 5% chance per cycle
    fault_duration_seconds: int = Field(default=60, ge=0)

    # Persistent Runtime State, Per-line date pointer so each line advances independently
    date_ptr: int = Field(default=0, description="Pointer to the current dataset date index")
    lot_counter: int = Field(default=0, description="Sequential lot ID counter")
    year_offset: int = Field(default=0, description="Years to add to base data once unique dates deplete")

def get_default_lines() -> Dict[str, LineConfig]:
    return  {
        "LINE_A": LineConfig(line_id="LINE_A", tester_id="TST-01", is_running=False),
        "LINE_B": LineConfig(line_id="LINE_B", tester_id="TST-02", is_running=False),
        "LINE_C": LineConfig(line_id="LINE_C", tester_id="TST-03", is_running=False),
    }

class SimulationConfig(BaseSettings):
    """Root config object — a dict of per-line configs."""
    lines: Dict[str, LineConfig] = Field(default_factory=get_default_lines)
 
    def get_line(self, line_id: str) -> LineConfig:
        return self.lines.get(line_id, LineConfig(line_id=line_id))
 
    def any_running(self) -> bool:
        return any(lc.is_running for lc in self.lines.values())

class StateStore:
    """Thread-safe SQLite-backed state store. Serialises config as JSON."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS simulation_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS line_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            line_id     TEXT NOT NULL,
            event_type  TEXT NOT NULL, -- STARTED | STOPPED | BATCH
            payload     TEXT,
            ts          TEXT NOT NULL
        );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        with self._conn() as conn:
            conn.executescript(self._SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)

    def get_config(self) -> SimulationConfig:

        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM simulation_state WHERE key='config'"
            ).fetchone()

        if not row:
            return SimulationConfig()

        try:
            return SimulationConfig.model_validate_json(row[0])
        except Exception:
            return SimulationConfig()

    def update_config(self, config: SimulationConfig) -> None:
        json_data = config.model_dump_json()

        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO simulation_state (key, value, updated_at) VALUES (?, ?, ?)",
                ("config", json_data, datetime.utcnow().isoformat()),
            )

    def update_line(self, line_id: str, line_config: LineConfig) -> None:
        config = self.get_config()
        config.lines[line_id] = line_config
        self.update_config(config)


    def log_event(self, line_id: str, event_type: str, payload: Optional[dict] = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO line_events (line_id, event_type, payload, ts) VALUES (?, ?, ?, ?)",
                (line_id, event_type, json.dumps(payload or {}), datetime.utcnow().isoformat()),
            )

    def get_events(self, line_id: Optional[str] = None, hours: int = 24) -> list:
        sql = """
            SELECT line_id, event_type, payload, ts FROM line_events
            WHERE ts >= datetime('now', ? )
        """
        params: list = [f"-{hours} hours"]
        if line_id:
            sql += " AND line_id = ?"
            params.append(line_id)
        sql += " ORDER BY ts DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            
        return [{"line_id": r[0], "event_type": r[1], "payload": json.loads(r[2]), "ts": r[3]} for r in rows]