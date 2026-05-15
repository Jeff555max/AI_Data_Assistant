from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from app.services.file_service import FileService, StoredFile


logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, file_service: FileService) -> None:
        self.file_service = file_service

    def analyze(self, stored_file: StoredFile) -> dict[str, Any]:
        if stored_file.kind == "table":
            return self._analyze_table(stored_file)
        if stored_file.kind == "image":
            return self._analyze_image(stored_file)
        return self._analyze_pdf(stored_file)

    def _analyze_table(self, stored_file: StoredFile) -> dict[str, Any]:
        dataframe = self.file_service.read_dataframe(stored_file)
        columns = self.file_service.describe_columns(dataframe)

        numeric_frame = dataframe.select_dtypes(include=[np.number])
        stats_records: list[dict[str, str]] = []
        if not numeric_frame.empty:
            for column in numeric_frame.columns:
                series = numeric_frame[column]
                stats_records.append(
                    {
                        "column": str(column),
                        "mean": self.file_service.format_value(series.mean(skipna=True)),
                        "median": self.file_service.format_value(series.median(skipna=True)),
                        "std": self.file_service.format_value(series.std(skipna=True)),
                        "min": self.file_service.format_value(series.min(skipna=True)),
                        "max": self.file_service.format_value(series.max(skipna=True)),
                    }
                )

        missing_summary = []
        total_rows = max(len(dataframe), 1)
        for column in dataframe.columns:
            missing_count = int(dataframe[column].isna().sum())
            if missing_count:
                missing_summary.append(
                    {
                        "column": str(column),
                        "missing": missing_count,
                        "percent": f"{(missing_count / total_rows) * 100:.1f}%",
                    }
                )

        insights = [
            f"Найдено {len(dataframe):,} строк и {len(dataframe.columns)} колонок.".replace(",", " "),
            "Числовая статистика рассчитана с игнорированием NaN, пропуски вынесены в отдельный блок.",
        ]
        if stats_records:
            widest_spread = max(stats_records, key=lambda item: self._to_float(item["std"]))
            insights.append(f"Наибольшая вариативность у колонки «{widest_spread['column']}».")

        category_candidates = [
            item for item in columns if item["kind"] in {"categorical", "text"} and item["missing"] < len(dataframe)
        ]
        if category_candidates:
            insights.append(
                f"Колонка «{category_candidates[0]['name']}» подходит для bar-chart сегментации."
            )

        logger.info("Completed tabular analysis for %s", stored_file.file_id)
        return {
            "kind": "table",
            "summary": {
                "rows": int(len(dataframe)),
                "columns": int(len(dataframe.columns)),
                "numeric_columns": int(len(numeric_frame.columns)),
                "missing_cells": int(dataframe.isna().sum().sum()),
            },
            "column_profile": columns,
            "stats": stats_records,
            "missing_summary": missing_summary,
            "insights": insights,
        }

    def _analyze_image(self, stored_file: StoredFile) -> dict[str, Any]:
        image = self.file_service.open_image(stored_file)
        array = np.array(image)
        grayscale = np.array(image.convert("L"))

        insights = [
            f"Разрешение изображения: {image.width}×{image.height}.",
            f"Средняя яркость: {grayscale.mean():.2f}.",
            "Для изображений доступны histogram, bar и line графики на основе пиксельных значений.",
        ]

        if array.ndim == 3:
            channel_names = list(image.getbands())
            stats = []
            for index, channel_name in enumerate(channel_names):
                channel = array[:, :, index]
                stats.append(
                    {
                        "channel": channel_name,
                        "mean": self.file_service.format_value(float(channel.mean())),
                        "median": self.file_service.format_value(float(np.median(channel))),
                        "std": self.file_service.format_value(float(channel.std())),
                        "min": self.file_service.format_value(int(channel.min())),
                        "max": self.file_service.format_value(int(channel.max())),
                    }
                )
        else:
            stats = [
                {
                    "channel": "L",
                    "mean": self.file_service.format_value(float(grayscale.mean())),
                    "median": self.file_service.format_value(float(np.median(grayscale))),
                    "std": self.file_service.format_value(float(grayscale.std())),
                    "min": self.file_service.format_value(int(grayscale.min())),
                    "max": self.file_service.format_value(int(grayscale.max())),
                }
            ]

        logger.info("Completed image analysis for %s", stored_file.file_id)
        return {
            "kind": "image",
            "summary": {
                "rows": image.height,
                "columns": image.width,
                "numeric_columns": len(stats),
                "missing_cells": 0,
            },
            "column_profile": [],
            "stats": stats,
            "missing_summary": [],
            "insights": insights,
        }

    def _analyze_pdf(self, stored_file: StoredFile) -> dict[str, Any]:
        pdf = self.file_service.read_pdf_text(stored_file)
        text = pdf["text"]
        words = re.findall(r"[\wА-Яа-яЁё]+", text.lower(), flags=re.UNICODE)
        meaningful_words = [
            word for word in words
            if len(word) > 3 and word not in self._pdf_stop_words()
        ]
        top_terms = Counter(meaningful_words).most_common(8)

        stats = [
            {"metric": "Страниц", "value": self.file_service.format_value(pdf["page_count"])},
            {"metric": "Слов", "value": self.file_service.format_value(pdf["word_count"])},
            {"metric": "Символов", "value": self.file_service.format_value(pdf["char_count"])},
            {"metric": "Источник текста", "value": self._pdf_text_source_label(pdf)},
        ]
        if pdf["ocr_used"]:
            stats.append(
                {
                    "metric": "OCR страниц",
                    "value": self.file_service.format_value(pdf["ocr_pages_read"]),
                }
            )
        if top_terms:
            stats.extend(
                {
                    "metric": f"Частый термин: {term}",
                    "value": self.file_service.format_value(count),
                }
                for term, count in top_terms[:5]
            )

        insights = [
            f"PDF содержит {pdf['page_count']} стр. и примерно {pdf['word_count']:,} слов.".replace(",", " "),
        ]
        if text.strip():
            if pdf["ocr_used"]:
                insights.append(
                    f"Текстовый слой не найден, поэтому выполнено OCR-распознавание "
                    f"{pdf['ocr_pages_read']} стр. Текст доступен для AI-анализа, отчёта и markdown-сводки."
                )
            else:
                insights.append("Текст успешно извлечён и доступен для AI-анализа, отчёта и markdown-сводки.")
        else:
            if pdf["ocr_error"]:
                insights.append(f"Из PDF не удалось получить текст. {pdf['ocr_error']}")
            else:
                insights.append("Из PDF не удалось получить текст ни из текстового слоя, ни через OCR.")
        if top_terms:
            terms = ", ".join(term for term, _ in top_terms[:5])
            insights.append(f"Самые частые содержательные термины: {terms}.")

        logger.info("Completed PDF analysis for %s", stored_file.file_id)
        return {
            "kind": "pdf",
            "summary": {
                "rows": int(pdf["page_count"]),
                "columns": int(pdf["word_count"]),
                "numeric_columns": 0,
                "missing_cells": 0,
                "pages": int(pdf["page_count"]),
                "words": int(pdf["word_count"]),
                "characters": int(pdf["char_count"]),
                "ocr_used": bool(pdf["ocr_used"]),
                "ocr_pages_read": int(pdf["ocr_pages_read"]),
            },
            "column_profile": [],
            "stats": stats,
            "missing_summary": [],
            "insights": insights,
        }

    def _to_float(self, value: str) -> float:
        try:
            return float(str(value).replace(" ", ""))
        except ValueError:
            return 0.0

    def _pdf_stop_words(self) -> set[str]:
        return {
            "для",
            "или",
            "это",
            "как",
            "при",
            "что",
            "если",
            "также",
            "the",
            "and",
            "with",
            "from",
            "this",
            "that",
            "или",
            "на",
            "по",
            "из",
            "под",
            "над",
            "без",
            "его",
            "её",
            "она",
            "они",
            "мы",
            "вы",
        }

    def _pdf_text_source_label(self, pdf: dict[str, Any]) -> str:
        if pdf["text_source"] == "ocr":
            return "OCR"
        if pdf["text_source"] == "text_layer":
            return "Текстовый слой"
        return "Не найден"
