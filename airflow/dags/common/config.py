from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Optional
from pathlib import Path

class ServiceConfig(BaseSettings):
    nats_endpoint:str = Field(default="nats://nats:4222")
    nats_subject:str = Field(default="secom.data.generated")
    nats_stream_name:str = Field(default="SECOM_PIPELINE")

    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=False
    )