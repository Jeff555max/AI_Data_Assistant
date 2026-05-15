from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.core.config import BASE_DIR, Settings
from app.services.chart_service import ChartService
from app.services.file_service import FileReadError, FileService, StoredFile


class ChartServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.settings = Settings(
            app_name="Test Data Assistant",
            app_host="127.0.0.1",
            app_port=8000,
            max_file_size="300MB",
            max_file_size_bytes=300 * 1024 * 1024,
            upload_dir=root / "uploads",
            output_dir=root / "outputs",
            storage_dir=root,
            templates_dir=BASE_DIR / "templates",
            static_dir=BASE_DIR / "static",
            log_level="INFO",
            openai_api_key=None,
            openai_model="gpt-4o",
            openai_max_history_messages=8,
        )
        self.file_service = FileService(self.settings)
        self.chart_service = ChartService(self.file_service, self.settings)
        sample_path = BASE_DIR / "examples" / "sample_sales_data.csv"
        self.stored_file = StoredFile(
            file_id="sample",
            original_name="sample_sales_data.csv",
            saved_name="sample_sales_data.csv",
            extension=".csv",
            content_type="text/csv",
            size_bytes=sample_path.stat().st_size,
            kind="table",
            created_at="2026-05-15T00:00:00+00:00",
            absolute_path=str(sample_path),
            relative_path="uploads/sample_sales_data.csv",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_generate_chart_resolves_columns_case_insensitively(self) -> None:
        chart = self.chart_service.generate_chart(
            self.stored_file,
            "bar",
            x_column="категория",
            y_column="прибыль",
        )

        self.assertTrue((self.settings.output_dir / chart["file_name"]).exists())

    def test_generate_chart_reports_missing_column(self) -> None:
        with self.assertRaisesRegex(FileReadError, "Доступные колонки"):
            self.chart_service.generate_chart(
                self.stored_file,
                "bar",
                x_column="Неизвестная колонка",
            )


if __name__ == "__main__":
    unittest.main()
