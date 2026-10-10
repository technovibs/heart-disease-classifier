from __future__ import annotations

from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str = getenv("APP_ENV", "development")
    api_host: str = getenv("API_HOST", "0.0.0.0")
    api_port: int = int(getenv("API_PORT", "8000"))
    mlflow_tracking_uri: str = getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


settings = Settings()
