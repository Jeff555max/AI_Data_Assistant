from __future__ import annotations

import logging
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
        return self._analyze_image(stored_file)

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

    def _to_float(self, value: str) -> float:
        try:
            return float(str(value).replace(" ", ""))
        except ValueError:
            return 0.0
