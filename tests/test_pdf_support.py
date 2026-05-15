from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import BASE_DIR, Settings
from app.services.analysis_service import AnalysisService
from app.services.chart_service import ChartService
from app.services.file_service import FileReadError, FileService, StoredFile


def build_text_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({escaped_text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    chunks = [b"%PDF-1.4\n"]
    offsets = []
    for index, payload in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii"))
        chunks.append(payload)
        chunks.append(b"\nendobj\n")

    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    for offset in offsets:
        chunks.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    chunks.append(
        f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return b"".join(chunks)


class PDFSupportTest(unittest.TestCase):
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
            pdf_ocr_enabled=True,
            pdf_ocr_languages="rus+eng",
            pdf_ocr_dpi=200,
            pdf_ocr_max_pages=20,
        )
        self.file_service = FileService(self.settings)
        self.analysis_service = AnalysisService(self.file_service)
        self.chart_service = ChartService(self.file_service, self.settings)
        self.pdf_path = root / "sample.pdf"
        self.pdf_path.write_bytes(
            build_text_pdf("PDF support works for analytics reports\nJan 100\nFeb 200\nMar 150")
        )
        self.stored_file = StoredFile(
            file_id="sample-pdf",
            original_name="sample.pdf",
            saved_name="sample.pdf",
            extension=".pdf",
            content_type="application/pdf",
            size_bytes=self.pdf_path.stat().st_size,
            kind="pdf",
            created_at="2026-05-15T00:00:00+00:00",
            absolute_path=str(self.pdf_path),
            relative_path="uploads/sample.pdf",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pdf_preview_extracts_text(self) -> None:
        preview = self.file_service.build_preview_context(self.stored_file)

        self.assertEqual(preview["kind"], "pdf")
        self.assertEqual(preview["page_count"], 1)
        self.assertTrue(preview["has_text"])
        self.assertFalse(preview["ocr_used"])
        self.assertEqual(preview["text_source"], "text_layer")
        self.assertIn("PDF support works", preview["text_excerpt"])

    def test_pdf_analysis_uses_document_metrics(self) -> None:
        analysis = self.analysis_service.analyze(self.stored_file)

        self.assertEqual(analysis["kind"], "pdf")
        self.assertEqual(analysis["summary"]["pages"], 1)
        self.assertGreaterEqual(analysis["summary"]["words"], 5)

    def test_pdf_chart_generation_uses_extracted_numbers(self) -> None:
        chart = self.chart_service.generate_chart(self.stored_file, "bar")

        self.assertTrue((self.settings.output_dir / chart["file_name"]).exists())
        self.assertIn("PDF", chart["description"])

    def test_pdf_without_text_layer_uses_ocr_fallback(self) -> None:
        scanned_path = Path(self.temp_dir.name) / "scanned.pdf"
        scanned_path.write_bytes(build_text_pdf(""))
        scanned_file = StoredFile(
            file_id="scanned-pdf",
            original_name="scanned.pdf",
            saved_name="scanned.pdf",
            extension=".pdf",
            content_type="application/pdf",
            size_bytes=scanned_path.stat().st_size,
            kind="pdf",
            created_at="2026-05-15T00:00:00+00:00",
            absolute_path=str(scanned_path),
            relative_path="uploads/scanned.pdf",
        )

        with patch.object(
            FileService,
            "_read_pdf_text_with_ocr",
            return_value={
                "text": "OCR распознал сканированный PDF",
                "pages_read": 1,
                "error": None,
                "cached": False,
            },
        ):
            preview = self.file_service.build_preview_context(scanned_file)

        self.assertTrue(preview["has_text"])
        self.assertTrue(preview["ocr_used"])
        self.assertEqual(preview["text_source"], "ocr")
        self.assertEqual(preview["ocr_pages_read"], 1)
        self.assertIn("OCR распознал", preview["text_excerpt"])

    def test_image_formats_open_and_ocr_text_can_drive_chart(self) -> None:
        from PIL import Image

        with patch.object(
            FileService,
            "read_image_text",
            return_value={
                "text": "",
                "char_count": 0,
                "word_count": 0,
                "ocr_used": False,
                "ocr_available": True,
                "ocr_error": None,
                "ocr_cached": False,
            },
        ):
            for extension, image_format in (
                (".png", "PNG"),
                (".jpg", "JPEG"),
                (".jpeg", "JPEG"),
                (".bmp", "BMP"),
                (".gif", "GIF"),
                (".webp", "WEBP"),
            ):
                image_path = Path(self.temp_dir.name) / f"infographic{extension}"
                Image.new("RGB", (240, 120), "white").save(image_path, image_format)
                stored_image = StoredFile(
                    file_id=f"image-{image_format.lower()}",
                    original_name=image_path.name,
                    saved_name=image_path.name,
                    extension=extension,
                    content_type=f"image/{image_format.lower()}",
                    size_bytes=image_path.stat().st_size,
                    kind="image",
                    created_at="2026-05-15T00:00:00+00:00",
                    absolute_path=str(image_path),
                    relative_path=f"uploads/{image_path.name}",
                )

                preview = self.file_service.build_preview_context(stored_image)
                self.assertEqual(preview["kind"], "image")
                self.assertGreater(preview["image_width"], 0)

        infographic = StoredFile(
            file_id="image-ocr-chart",
            original_name="infographic.png",
            saved_name="infographic.png",
            extension=".png",
            content_type="image/png",
            size_bytes=(Path(self.temp_dir.name) / "infographic.png").stat().st_size,
            kind="image",
            created_at="2026-05-15T00:00:00+00:00",
            absolute_path=str(Path(self.temp_dir.name) / "infographic.png"),
            relative_path="uploads/infographic.png",
        )
        with patch.object(
            FileService,
            "read_image_text",
            return_value={
                "text": "Product A 120\nProduct B 80\nProduct C 160",
                "char_count": 38,
                "word_count": 9,
                "ocr_used": True,
                "ocr_available": True,
                "ocr_error": None,
                "ocr_cached": False,
            },
        ):
            chart = self.chart_service.generate_chart(infographic, "bar")

        self.assertTrue((self.settings.output_dir / chart["file_name"]).exists())
        self.assertIn("OCR изображения", chart["description"])


if __name__ == "__main__":
    unittest.main()
