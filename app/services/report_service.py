from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

from app.core.config import Settings, get_settings
from app.services.file_service import FileService, StoredFile


logger = logging.getLogger(__name__)


class ReportService:
    def __init__(self, file_service: FileService, settings: Settings | None = None) -> None:
        self.file_service = file_service
        self.settings = settings or get_settings()

    def generate_report(
        self,
        stored_file: StoredFile,
        analysis: dict[str, Any],
        charts: list[dict[str, Any]],
        preview_context: dict[str, Any],
    ) -> dict[str, str]:
        self.file_service.ensure_storage()
        report_name = self._build_report_name(stored_file.file_id)
        report_path = self.settings.output_dir / report_name

        document = Document()
        title = document.add_heading("Data Assistant Report", level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        document.add_paragraph(f"Исходный файл: {stored_file.original_name}")
        document.add_paragraph(f"Тип: {stored_file.kind}")
        document.add_paragraph(f"Сформирован: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")

        document.add_heading("Краткое резюме", level=1)
        summary_table = document.add_table(rows=1, cols=2)
        summary_table.style = "Light Grid Accent 1"
        header_cells = summary_table.rows[0].cells
        header_cells[0].text = "Метрика"
        header_cells[1].text = "Значение"
        for label, value in {
            "Rows / Height": analysis["summary"]["rows"],
            "Columns / Width": analysis["summary"]["columns"],
            "Numeric metrics": analysis["summary"]["numeric_columns"],
            "Missing cells": analysis["summary"]["missing_cells"],
        }.items():
            row = summary_table.add_row().cells
            row[0].text = str(label)
            row[1].text = str(value)

        document.add_heading("Выводы", level=1)
        for insight in analysis["insights"]:
            document.add_paragraph(insight, style="List Bullet")

        if analysis["stats"]:
            document.add_heading("Статистика", level=1)
            headers = list(analysis["stats"][0].keys())
            stats_table = document.add_table(rows=1, cols=len(headers))
            stats_table.style = "Light Shading Accent 1"
            for index, header in enumerate(headers):
                stats_table.rows[0].cells[index].text = str(header)
            for item in analysis["stats"]:
                cells = stats_table.add_row().cells
                for index, header in enumerate(headers):
                    cells[index].text = str(item[header])

        if analysis["missing_summary"]:
            document.add_heading("Пропуски", level=1)
            missing_table = document.add_table(rows=1, cols=3)
            missing_table.style = "Light Grid Accent 2"
            for index, header in enumerate(["Column", "Missing", "Percent"]):
                missing_table.rows[0].cells[index].text = header
            for item in analysis["missing_summary"]:
                cells = missing_table.add_row().cells
                cells[0].text = item["column"]
                cells[1].text = str(item["missing"])
                cells[2].text = item["percent"]

        document.add_heading("Preview", level=1)
        if stored_file.kind == "table":
            headers = preview_context["table_headers"][:6]
            preview_table = document.add_table(rows=1, cols=len(headers))
            preview_table.style = "Table Grid"
            for index, header in enumerate(headers):
                preview_table.rows[0].cells[index].text = str(header)
            for row_values in preview_context["table_rows"][:10]:
                cells = preview_table.add_row().cells
                for index, value in enumerate(row_values[: len(headers)]):
                    cells[index].text = str(value)
        else:
            document.add_paragraph(
                f"Изображение {preview_context['image_width']}×{preview_context['image_height']} ({preview_context['image_mode']})"
            )

        if charts:
            document.add_heading("Графики", level=1)
            for chart in charts:
                document.add_paragraph(chart.get("description", f"Сгенерированный график: {chart['file_name']}"))
                document.add_picture(
                    str(self.settings.output_dir / chart["file_name"]),
                    width=Inches(6.3),
                )

        document.save(report_path)
        logger.info("Generated report %s for %s", report_name, stored_file.file_id)
        return {
            "file_name": report_name,
            "storage_url": f"/storage/outputs/{report_name}",
            "download_url": f"/download/{report_name}",
        }

    def _build_report_name(self, file_id: str) -> str:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
        return f"{file_id}__report__{timestamp}.docx"
