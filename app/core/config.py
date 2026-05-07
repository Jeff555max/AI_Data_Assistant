from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*(B|KB|MB|GB)?\s*$", re.IGNORECASE)
SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def parse_size_to_bytes(raw_value: str) -> int:
    match = SIZE_PATTERN.match(raw_value or "")
    if not match:
        raise ValueError(f"Invalid size format: {raw_value!r}")
    amount = int(match.group(1))
    unit = (match.group(2) or "B").upper()
    return amount * SIZE_UNITS[unit]


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_host: str
    app_port: int
    max_file_size: str
    max_file_size_bytes: int
    upload_dir: Path
    output_dir: Path
    storage_dir: Path
    templates_dir: Path
    static_dir: Path
    log_level: str
    openai_api_key: str | None
    openai_model: str
    openai_max_history_messages: int


@lru_cache
def get_settings() -> Settings:
    upload_dir = BASE_DIR / os.getenv("UPLOAD_DIR", "storage/uploads")
    output_dir = BASE_DIR / os.getenv("OUTPUT_DIR", "storage/outputs")
    return Settings(
        app_name=os.getenv("APP_NAME", "Data Assistant"),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        max_file_size=os.getenv("MAX_FILE_SIZE", "10MB"),
        max_file_size_bytes=parse_size_to_bytes(os.getenv("MAX_FILE_SIZE", "10MB")),
        upload_dir=upload_dir,
        output_dir=output_dir,
        storage_dir=BASE_DIR / "storage",
        templates_dir=BASE_DIR / "templates",
        static_dir=BASE_DIR / "static",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        openai_max_history_messages=int(os.getenv("OPENAI_MAX_HISTORY_MESSAGES", "8")),
    )
