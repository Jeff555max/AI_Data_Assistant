from __future__ import annotations

import json
import logging
import mimetypes
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)


class FileServiceError(Exception):
    """Base file service error."""


class UnsupportedFileError(FileServiceError):
    """Raised when the file extension is not supported."""


class FileTooLargeError(FileServiceError):
    """Raised when the file is larger than the configured limit."""


class EmptyFileError(FileServiceError):
    """Raised when the uploaded file is empty."""


class FileReadError(FileServiceError):
    """Raised when the file cannot be parsed."""


SUPPORTED_EXTENSIONS = {
    ".csv": "table",
    ".xlsx": "table",
    ".xls": "table",
    ".json": "table",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
    ".gif": "image",
    ".webp": "image",
}


def _safe_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename.strip()) or "upload"


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass
class StoredFile:
    file_id: str
    original_name: str
    saved_name: str
    extension: str
    content_type: str
    size_bytes: int
    kind: str
    created_at: str
    absolute_path: str
    relative_path: str

    @property
    def path(self) -> Path:
        return Path(self.absolute_path)


class FileService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def ensure_storage(self) -> None:
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile) -> StoredFile:
        self.ensure_storage()

        original_name = upload.filename or "upload"
        extension = Path(original_name).suffix.lower()
        kind = SUPPORTED_EXTENSIONS.get(extension)
        if not kind:
            raise UnsupportedFileError(
                "Поддерживаются CSV, Excel, JSON и изображения PNG/JPG/JPEG/BMP/GIF/WEBP."
            )

        file_id = uuid4().hex
        safe_name = _safe_name(Path(original_name).name)
        saved_name = f"{file_id}_{safe_name}"
        destination = self.settings.upload_dir / saved_name
        total_size = 0

        try:
            with destination.open("wb") as buffer:
                while chunk := await upload.read(1024 * 1024):
                    total_size += len(chunk)
                    if total_size > self.settings.max_file_size_bytes:
                        raise FileTooLargeError(
                            f"Размер файла превышает лимит {self.settings.max_file_size}."
                        )
                    buffer.write(chunk)
        except FileTooLargeError:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        if total_size == 0:
            destination.unlink(missing_ok=True)
            raise EmptyFileError("Файл пустой. Загрузите файл с данными.")

        content_type = upload.content_type or mimetypes.guess_type(destination.name)[0] or "application/octet-stream"
        stored_file = StoredFile(
            file_id=file_id,
            original_name=original_name,
            saved_name=saved_name,
            extension=extension,
            content_type=content_type,
            size_bytes=total_size,
            kind=kind,
            created_at=_utc_now(),
            absolute_path=str(destination),
            relative_path=f"uploads/{saved_name}",
        )
        self._write_metadata(stored_file)
        logger.info("Saved upload %s (%s)", stored_file.file_id, stored_file.original_name)
        return stored_file

    def get_file(self, file_id: str) -> StoredFile:
        metadata_path = self.settings.upload_dir / f"{file_id}.json"
        if not metadata_path.exists():
            raise FileReadError("Файл не найден или срок хранения истек.")
        return StoredFile(**json.loads(metadata_path.read_text(encoding="utf-8")))

    def read_dataframe(self, stored_file: StoredFile) -> pd.DataFrame:
        path = stored_file.path
        try:
            if stored_file.extension == ".csv":
                dataframe = pd.read_csv(path, low_memory=False)
            elif stored_file.extension in {".xlsx", ".xls"}:
                dataframe = pd.read_excel(path)
            elif stored_file.extension == ".json":
                dataframe = self._read_json(path)
            else:
                raise FileReadError("Этот файл не является табличным.")
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            logger.exception("Failed to read data file %s", stored_file.path)
            raise FileReadError("Не удалось прочитать файл. Проверьте формат и содержимое.") from exc

        if dataframe.empty:
            raise EmptyFileError("Таблица не содержит строк для анализа.")
        return dataframe

    def open_image(self, stored_file: StoredFile) -> Image.Image:
        try:
            image = Image.open(stored_file.path)
            image.load()
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            logger.exception("Failed to open image %s", stored_file.path)
            raise FileReadError("Не удалось прочитать изображение.") from exc
        return image

    def build_preview_context(self, stored_file: StoredFile, rows: int = 15) -> dict[str, Any]:
        base_context: dict[str, Any] = {
            "file": stored_file,
            "file_size_kb": round(stored_file.size_bytes / 1024, 2),
            "storage_url": f"/storage/{stored_file.relative_path}",
        }

        if stored_file.kind == "table":
            dataframe = self.read_dataframe(stored_file)
            columns = self.describe_columns(dataframe)
            preview = dataframe.head(rows)
            table_rows = [
                [self.format_value(value) for value in row]
                for row in preview.itertuples(index=False, name=None)
            ]
            numeric_columns = [item["name"] for item in columns if item["kind"] == "numeric"]
            dimension_columns = [item["name"] for item in columns if item["kind"] in {"categorical", "datetime"}]
            all_columns = [str(column) for column in dataframe.columns]
            return {
                **base_context,
                "kind": "table",
                "row_count": int(len(dataframe)),
                "column_count": int(len(dataframe.columns)),
                "columns": columns,
                "table_headers": all_columns,
                "table_rows": table_rows,
                "numeric_columns": numeric_columns,
                "dimension_columns": dimension_columns or all_columns,
                "recommended_x": (dimension_columns or all_columns or [""])[0],
                "recommended_y": (numeric_columns or all_columns or [""])[0],
            }

        image = self.open_image(stored_file)
        array = np.array(image)
        channel_means = (
            np.round(array.reshape(-1, array.shape[-1]).mean(axis=0), 2).tolist()
            if array.ndim == 3
            else []
        )
        return {
            **base_context,
            "kind": "image",
            "image_width": image.width,
            "image_height": image.height,
            "image_mode": image.mode,
            "image_format": image.format or stored_file.extension.replace(".", "").upper(),
            "channel_means": channel_means,
        }

    def describe_columns(self, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        descriptions: list[dict[str, Any]] = []
        for column in dataframe.columns:
            series = dataframe[column]
            non_null = series.dropna()
            descriptions.append(
                {
                    "name": str(column),
                    "dtype": str(series.dtype),
                    "kind": self.detect_column_kind(series),
                    "missing": int(series.isna().sum()),
                    "sample": self.format_value(non_null.iloc[0]) if not non_null.empty else "—",
                }
            )
        return descriptions

    def detect_column_kind(self, series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        non_null = series.dropna()
        if non_null.empty:
            return "categorical"

        sample_values = non_null.astype(str).head(10)
        looks_date_like = (
            sample_values.str.contains(r"\d", regex=True).mean() >= 0.6
            and sample_values.str.contains(r"[-/:T]", regex=True).mean() >= 0.6
        )
        if looks_date_like:
            parsed_dates = pd.to_datetime(non_null, errors="coerce")
            if parsed_dates.notna().mean() >= 0.8:
                return "datetime"

        unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
        return "categorical" if unique_ratio < 0.5 else "text"

    def get_output_artifacts(self, file_id: str) -> dict[str, list[dict[str, str]]]:
        charts: list[dict[str, str]] = []
        reports: list[dict[str, str]] = []

        for artifact in sorted(self.settings.output_dir.glob(f"{file_id}__*"), reverse=True):
            record = {
                "name": artifact.name,
                "file_name": artifact.name,
                "storage_url": f"/storage/outputs/{artifact.name}",
                "download_url": f"/download/{artifact.name}",
            }
            if artifact.suffix.lower() == ".png":
                charts.append(record)
            elif artifact.suffix.lower() in {".docx", ".pdf"}:
                reports.append(record)

        return {"charts": charts, "reports": reports}

    def format_value(self, value: Any) -> str:
        if value is None:
            return "—"
        if not isinstance(value, str) and pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:,.3f}".replace(",", " ")
        if isinstance(value, (np.integer, int)):
            return f"{int(value):,}".replace(",", " ")
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)

    def _write_metadata(self, stored_file: StoredFile) -> None:
        metadata_path = self.settings.upload_dir / f"{stored_file.file_id}.json"
        metadata_path.write_text(
            json.dumps(asdict(stored_file), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_json(self, path: Path) -> pd.DataFrame:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            if all(isinstance(value, list) for value in payload.values()):
                return pd.DataFrame(payload)
            return pd.json_normalize(payload)
        raise FileReadError("JSON должен содержать объект или массив объектов.")
