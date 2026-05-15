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

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    PdfReader = None  # type: ignore[assignment]

try:
    import fitz
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    fitz = None  # type: ignore[assignment]

try:
    import pytesseract
    from pytesseract import TesseractError, TesseractNotFoundError
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    pytesseract = None  # type: ignore[assignment]
    TesseractError = RuntimeError  # type: ignore[assignment]
    TesseractNotFoundError = RuntimeError  # type: ignore[assignment]


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
    ".pdf": "pdf",
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
                "Поддерживаются CSV, Excel, JSON, PDF и изображения PNG/JPG/JPEG/BMP/GIF/WEBP."
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

    def read_pdf_text(self, stored_file: StoredFile, max_pages: int | None = None) -> dict[str, Any]:
        if PdfReader is None:
            raise FileReadError(
                "Для чтения PDF нужен пакет `pypdf`. Выполните `pip install -r requirements.txt`."
            )

        if stored_file.extension != ".pdf":
            raise FileReadError("Этот файл не является PDF.")

        try:
            reader = PdfReader(str(stored_file.path))
            if reader.is_encrypted:
                decrypt_result = reader.decrypt("")
                if decrypt_result == 0:
                    raise FileReadError("PDF защищён паролем. Загрузите файл без пароля.")

            page_count = len(reader.pages)
            pages_to_read = page_count if max_pages is None else min(page_count, max(max_pages, 0))
            page_texts = []
            for page_number in range(pages_to_read):
                text = reader.pages[page_number].extract_text() or ""
                page_texts.append(text.strip())
        except FileReadError:
            raise
        except Exception as exc:
            logger.exception("Failed to read PDF %s", stored_file.path)
            raise FileReadError("Не удалось прочитать PDF. Проверьте формат и защиту файла.") from exc

        text = "\n\n".join(item for item in page_texts if item)
        text_source = "text_layer" if self._has_meaningful_pdf_text(text) else "none"
        ocr_result = {
            "text": "",
            "pages_read": 0,
            "error": None,
            "cached": False,
        }
        if text_source == "none" and self.settings.pdf_ocr_enabled:
            ocr_result = self._read_pdf_text_with_ocr(stored_file, page_count)
            if ocr_result["text"].strip():
                text = ocr_result["text"]
                text_source = "ocr"

        words = re.findall(r"[\wА-Яа-яЁё]+", text, flags=re.UNICODE)
        return {
            "page_count": page_count,
            "pages_read": pages_to_read,
            "text": text,
            "char_count": len(text),
            "word_count": len(words),
            "text_source": text_source,
            "ocr_used": text_source == "ocr",
            "ocr_enabled": self.settings.pdf_ocr_enabled,
            "ocr_available": fitz is not None and pytesseract is not None,
            "ocr_error": ocr_result["error"],
            "ocr_pages_read": ocr_result["pages_read"],
            "ocr_page_limit": self.settings.pdf_ocr_max_pages,
            "ocr_cached": ocr_result["cached"],
        }

    def _read_pdf_text_with_ocr(self, stored_file: StoredFile, page_count: int) -> dict[str, Any]:
        if fitz is None or pytesseract is None:
            return {
                "text": "",
                "pages_read": 0,
                "error": "OCR недоступен: установите `PyMuPDF`, `pytesseract` и Tesseract.",
                "cached": False,
            }

        cached = self._read_ocr_cache(stored_file)
        if cached:
            return {**cached, "cached": True}

        pages_to_read = min(page_count, max(self.settings.pdf_ocr_max_pages, 0))
        if pages_to_read == 0:
            return {
                "text": "",
                "pages_read": 0,
                "error": "OCR отключён лимитом `PDF_OCR_MAX_PAGES=0`.",
                "cached": False,
            }

        scale = max(self.settings.pdf_ocr_dpi, 72) / 72
        matrix = fitz.Matrix(scale, scale)
        page_texts: list[str] = []

        try:
            with fitz.open(str(stored_file.path)) as document:
                for page_number in range(min(pages_to_read, len(document))):
                    page = document.load_page(page_number)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    text = pytesseract.image_to_string(
                        image,
                        lang=self.settings.pdf_ocr_languages,
                    )
                    page_texts.append(text.strip())
        except TesseractNotFoundError:
            return {
                "text": "",
                "pages_read": 0,
                "error": "OCR недоступен: бинарный файл Tesseract не найден.",
                "cached": False,
            }
        except TesseractError as exc:
            logger.exception("Tesseract OCR failed for %s", stored_file.path)
            return {
                "text": "",
                "pages_read": len(page_texts),
                "error": f"OCR не удалось выполнить: {exc}.",
                "cached": False,
            }
        except Exception as exc:
            logger.exception("Failed to OCR PDF %s", stored_file.path)
            return {
                "text": "",
                "pages_read": len(page_texts),
                "error": "OCR не удалось выполнить. Проверьте PDF и настройки распознавания.",
                "cached": False,
            }

        text = "\n\n".join(item for item in page_texts if item)
        result = {
            "text": text,
            "pages_read": len(page_texts),
            "error": None if text.strip() else "OCR выполнен, но текст не распознан.",
        }
        self._write_ocr_cache(stored_file, result)
        return {**result, "cached": False}

    def _has_meaningful_pdf_text(self, text: str) -> bool:
        words = re.findall(r"[\wА-Яа-яЁё]+", text, flags=re.UNICODE)
        return len(words) >= 5

    def _read_ocr_cache(self, stored_file: StoredFile) -> dict[str, Any] | None:
        cache_path = self._ocr_cache_path(stored_file)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        cache_key = self._ocr_cache_key(stored_file)
        if payload.get("cache_key") != cache_key:
            return None

        return {
            "text": str(payload.get("text", "")),
            "pages_read": int(payload.get("pages_read", 0)),
            "error": payload.get("error"),
        }

    def _write_ocr_cache(self, stored_file: StoredFile, result: dict[str, Any]) -> None:
        self.ensure_storage()
        payload = {
            "cache_key": self._ocr_cache_key(stored_file),
            "text": result["text"],
            "pages_read": result["pages_read"],
            "error": result["error"],
            "created_at": _utc_now(),
        }
        self._ocr_cache_path(stored_file).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ocr_cache_path(self, stored_file: StoredFile) -> Path:
        return self.settings.upload_dir / f"{stored_file.file_id}.ocr.json"

    def _ocr_cache_key(self, stored_file: StoredFile) -> dict[str, Any]:
        return {
            "file_id": stored_file.file_id,
            "size_bytes": stored_file.size_bytes,
            "languages": self.settings.pdf_ocr_languages,
            "dpi": self.settings.pdf_ocr_dpi,
            "max_pages": self.settings.pdf_ocr_max_pages,
        }

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

        if stored_file.kind == "pdf":
            pdf = self.read_pdf_text(stored_file)
            excerpt = pdf["text"][:4000]
            return {
                **base_context,
                "kind": "pdf",
                "page_count": pdf["page_count"],
                "preview_pages": pdf["pages_read"],
                "word_count": pdf["word_count"],
                "char_count": pdf["char_count"],
                "text_excerpt": excerpt,
                "has_text": bool(excerpt.strip()),
                "text_source": pdf["text_source"],
                "ocr_used": pdf["ocr_used"],
                "ocr_enabled": pdf["ocr_enabled"],
                "ocr_available": pdf["ocr_available"],
                "ocr_error": pdf["ocr_error"],
                "ocr_pages_read": pdf["ocr_pages_read"],
                "ocr_page_limit": pdf["ocr_page_limit"],
                "ocr_cached": pdf["ocr_cached"],
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
