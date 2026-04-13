from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Optional
from pathlib import Path

class ServiceConfig(BaseSettings):
    nats_endpoint:str = Field(default="nats://nats:4222")
    nats_subject:str = Field(default="secom.data.generated")
    
    @property
    def minio_access_key(self):
        return Path(self.minio_access_key_file).read_text().strip()

    @property
    def minio_secret_key(self):
        return Path(self.minio_secret_key_file).read_text().strip()

    @property
    def catalog_uri(self):
        return f"jdbc:postgresql://postgres:5432/{catalog_name}"

    @property
    def catalog_password(self):
        return Path(self.airflow_db_password_file).read_text().strip()

    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=False
    )