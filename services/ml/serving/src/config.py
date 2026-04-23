from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Optional
from pathlib import Path

class ServiceConfig(BaseSettings):
    minio_endpoint:str = Field(default="http://minio:9000")
    minio_warehouse:str = Field(default="s3a://warehouse")
    minio_access_key_file:str
    minio_secret_key_file:str

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