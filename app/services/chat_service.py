from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings, get_settings
from app.services.ai_service import AIService, AIServiceConfigurationError, AIServiceRequestError
from app.services.analysis_service import AnalysisService
from app.services.chart_service import ChartService
from app.services.export_service import ExportService
from app.services.file_service import FileService, FileServiceError, StoredFile
from app.services.report_service import ReportService


logger = logging.getLogger(__name__)


class ChatNotFoundError(FileNotFoundError):
    """Raised when a chat transcript does not exist."""


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _time_label(iso_value: str) -> str:
    return datetime.fromisoformat(iso_value).astimezone(UTC).strftime("%H:%M UTC")


class ChatService:
    def __init__(
        self,
        file_service: FileService,
        analysis_service: AnalysisService,
        chart_service: ChartService,
        report_service: ReportService,
        export_service: ExportService,
        ai_service: AIService,
        settings: Settings | None = None,
    ) -> None:
        self.file_service = file_service
        self.analysis_service = analysis_service
        self.chart_service = chart_service
        self.report_service = report_service
        self.export_service = export_service
        self.ai_service = ai_service
        self.settings = settings or get_settings()
        self.chat_dir = self.settings.storage_dir / "chats"

    def ensure_storage(self) -> None:
        self.chat_dir.mkdir(parents=True, exist_ok=True)

    def create_conversation(self) -> dict[str, Any]:
        self.ensure_storage()
        conversation_id = uuid4().hex
        created_at = _utc_now()
        greeting = (
            "Я подключён к OpenAI и работаю как AI-ассистент по данным. "
            "Можно отправить обычный текст, загрузить таблицу или изображение, попросить анализ, график, отчёт или сохранить выводы в файл."
        )
        conversation = {
            "conversation_id": conversation_id,
            "title": "AI Data Chat",
            "created_at": created_at,
            "updated_at": created_at,
            "active_file_id": None,
            "files": [],
            "messages": [self._new_message("assistant", greeting)],
        }
        self._save_conversation(conversation)
        return conversation

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        self.ensure_storage()
        path = self._chat_path(conversation_id)
        if not path.exists():
            raise ChatNotFoundError(f"Chat {conversation_id} not found.")
        return json.loads(path.read_text(encoding="utf-8"))

    async def process_turn(
        self,
        conversation_id: str,
        message_text: str,
        upload: UploadFile | None = None,
    ) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        text = (message_text or "").strip()
        uploaded_file: StoredFile | None = None
        user_attachments: list[dict[str, Any]] = []

        if upload and upload.filename:
            try:
                uploaded_file = await self.file_service.save_upload(upload)
            except FileServiceError as exc:
                if text:
                    conversation["messages"].append(self._new_message("user", text))
                conversation["messages"].append(
                    self._new_message("assistant", f"Не удалось обработать загрузку: {exc}")
                )
                return self._persist(conversation)

            conversation["files"] = self._upsert_file_record(conversation.get("files", []), uploaded_file)
            conversation["active_file_id"] = uploaded_file.file_id
            user_attachments.append(self._file_chip(uploaded_file))

        if not text and not user_attachments:
            conversation["messages"].append(
                self._new_message("assistant", "Сообщение пустое. Напишите запрос или прикрепите файл.")
            )
            return self._persist(conversation)

        user_text = text or "Загрузил файл для обработки."
        conversation["messages"].append(self._new_message("user", user_text, attachments=user_attachments))

        active_file = uploaded_file or self._get_active_file(conversation)
        raw_preview_context = self.file_service.build_preview_context(active_file) if active_file else None
        active_file_context = self._build_ai_file_context(active_file) if active_file else None

        try:
            plan = self.ai_service.plan_response(
                conversation_messages=conversation.get("messages", []),
                user_text=user_text,
                active_file=active_file,
                active_file_context=active_file_context,
            )
            assistant_message = self._new_message("assistant", plan.assistant_message)
            if uploaded_file:
                self._attach_uploaded_file(assistant_message, uploaded_file, raw_preview_context)
            self._apply_actions(
                assistant_message=assistant_message,
                actions=plan.actions,
                conversation=conversation,
                active_file=active_file,
                preview_context=raw_preview_context,
            )
            conversation["messages"].append(assistant_message)
            return self._persist(conversation)
        except AIServiceConfigurationError as exc:
            logger.warning("OpenAI is not configured: %s", exc)
            assistant_message = self._new_message(
                "assistant",
                f"OpenAI сейчас не настроен: {exc} Добавьте ключ в `.env`, и чат начнёт отвечать через модель.",
            )
            if uploaded_file:
                self._attach_uploaded_file(assistant_message, uploaded_file, raw_preview_context)
            conversation["messages"].append(assistant_message)
            return self._persist(conversation)
        except AIServiceRequestError as exc:
            logger.warning("OpenAI request failed, switching to local fallback: %s", exc)
            fallback_message = self._handle_local_prompt(conversation, user_text, active_file)
            fallback_message["text"] = (
                "OpenAI временно недоступен, поэтому я выполнил резервный локальный сценарий.\n\n"
                + fallback_message["text"]
            )
            if uploaded_file and not fallback_message.get("preview"):
                self._attach_uploaded_file(fallback_message, uploaded_file, raw_preview_context)
            conversation["messages"].append(fallback_message)
            return self._persist(conversation)

    def activate_file(self, conversation_id: str, file_id: str) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        file_record = next((item for item in conversation.get("files", []) if item["file_id"] == file_id), None)
        if not file_record:
            conversation["messages"].append(
                self._new_message("assistant", "Не нашёл этот файл в текущем чате.")
            )
            return self._persist(conversation)

        conversation["active_file_id"] = file_id
        conversation["messages"].append(
            self._new_message(
                "assistant",
                f"Сделал активным файл «{file_record['original_name']}». Теперь новые AI-запросы будут применяться к нему.",
            )
        )
        return self._persist(conversation)

    def build_page_context(self, request: Any, conversation: dict[str, Any]) -> dict[str, Any]:
        active_preview = None
        active_file_id = conversation.get("active_file_id")
        if active_file_id:
            try:
                active_preview = self.file_service.build_preview_context(self.file_service.get_file(active_file_id))
            except FileServiceError:
                active_preview = None

        files = [{**item, "is_active": item["file_id"] == active_file_id} for item in reversed(conversation.get("files", []))]
        return {
            "request": request,
            "page_title": "AI Data Chat",
            "conversation": conversation,
            "messages": conversation.get("messages", []),
            "active_preview": active_preview,
            "files": files,
            "saved_artifacts": self._collect_saved_artifacts(conversation),
            "ai_enabled": self.ai_service.enabled,
            "ai_model": self.ai_service.model_name,
        }

    def _apply_actions(
        self,
        *,
        assistant_message: dict[str, Any],
        actions: list[dict[str, Any]],
        conversation: dict[str, Any],
        active_file: StoredFile | None,
        preview_context: dict[str, Any] | None,
    ) -> None:
        if not active_file:
            return

        notes: list[str] = []
        analysis_cache: dict[str, Any] | None = None
        preview_cache = preview_context

        for action in actions:
            action_type = action["type"]
            try:
                if action_type == "preview":
                    if preview_cache:
                        assistant_message["preview"] = self._compact_preview(preview_cache)
                elif action_type == "analyze":
                    if analysis_cache is None:
                        analysis_cache = self.analysis_service.analyze(active_file)
                    assistant_message["analysis"] = self._compact_analysis(analysis_cache)
                elif action_type == "generate_chart":
                    chart = self.chart_service.generate_chart(
                        active_file,
                        action.get("chart_type", "bar"),
                        x_column=action.get("x_column"),
                        y_column=action.get("y_column"),
                    )
                    assistant_message["attachments"].append(self._artifact_chip(chart, "chart"))
                elif action_type == "generate_report":
                    if preview_cache is None:
                        preview_cache = self.file_service.build_preview_context(active_file)
                    if analysis_cache is None:
                        analysis_cache = self.analysis_service.analyze(active_file)
                    chart_records = self.file_service.get_output_artifacts(active_file.file_id)["charts"]
                    if not chart_records:
                        self.chart_service.generate_default_charts(active_file)
                        chart_records = self.file_service.get_output_artifacts(active_file.file_id)["charts"]
                    report = self.report_service.generate_report(
                        active_file,
                        analysis_cache,
                        chart_records[:3],
                        preview_cache,
                    )
                    assistant_message["attachments"].append(self._artifact_chip(report, "report"))
                elif action_type == "save_summary":
                    if analysis_cache is None:
                        analysis_cache = self.analysis_service.analyze(active_file)
                    charts = self.file_service.get_output_artifacts(active_file.file_id)["charts"]
                    export_record = self.export_service.generate_markdown_summary(
                        conversation["conversation_id"],
                        conversation.get("messages", []),
                        stored_file=active_file,
                        analysis=analysis_cache,
                        charts=charts[:3],
                    )
                    assistant_message["attachments"].append(self._artifact_chip(export_record, "export"))
            except FileServiceError as exc:
                notes.append(f"Не удалось выполнить действие `{action_type}`: {exc}")

        if notes:
            assistant_message["text"] = assistant_message["text"].rstrip() + "\n\n" + "\n".join(notes)

    def _attach_uploaded_file(
        self,
        assistant_message: dict[str, Any],
        stored_file: StoredFile,
        preview_context: dict[str, Any] | None,
    ) -> None:
        assistant_message["attachments"].append(self._stored_file_attachment(stored_file))
        if preview_context:
            assistant_message["preview"] = self._compact_preview(preview_context)

    def _build_ai_file_context(self, stored_file: StoredFile | None) -> dict[str, Any] | None:
        if not stored_file:
            return None

        preview = self.file_service.build_preview_context(stored_file)
        context: dict[str, Any] = {
            "file_id": stored_file.file_id,
            "original_name": stored_file.original_name,
            "kind": stored_file.kind,
            "created_at": stored_file.created_at,
        }

        if preview["kind"] == "table":
            context.update(
                {
                    "preview_kind": "table",
                    "row_count": preview["row_count"],
                    "column_count": preview["column_count"],
                    "headers": preview["table_headers"][:10],
                    "rows": preview["table_rows"][:6],
                    "columns": preview["columns"][:10],
                    "numeric_columns": preview["numeric_columns"][:10],
                    "dimension_columns": preview["dimension_columns"][:10],
                }
            )
            analysis = self.analysis_service.analyze(stored_file)
            context["analysis_summary"] = analysis["summary"]
            context["analysis_stats"] = analysis["stats"][:6]
            context["insights"] = analysis["insights"][:5]
            return context

        context.update(
            {
                "preview_kind": "image",
                "width": preview["image_width"],
                "height": preview["image_height"],
                "mode": preview["image_mode"],
                "format": preview["image_format"],
                "channel_means": preview["channel_means"],
                "storage_url": preview["storage_url"],
            }
        )
        return context

    def _handle_local_prompt(
        self,
        conversation: dict[str, Any],
        text: str,
        active_file: StoredFile | None,
    ) -> dict[str, Any]:
        normalized = text.lower()
        action = self._detect_action(normalized)

        if action == "help":
            if not active_file:
                return self._new_message(
                    "assistant",
                    "Пока нет активного файла. Загрузите CSV, Excel, JSON или изображение. После этого можно попросить анализ, график, отчёт или сохранение в файл.",
                )

            return self._new_message(
                "assistant",
                "Для активного файла могу показать предпросмотр, собрать анализ, построить `histogram`, `bar` или `line`, сделать DOCX-отчёт и сохранить краткую markdown-сводку. Для выбора колонок напишите, например: `Построй line chart x: date y: revenue`.",
            )

        if not active_file:
            return self._new_message(
                "assistant",
                "Сначала нужен файл или изображение. После загрузки я смогу строить графики и сохранять результаты.",
            )

        if action == "preview":
            preview = self.file_service.build_preview_context(active_file)
            return self._new_message(
                "assistant",
                f"Показываю текущий файл «{active_file.original_name}».",
                preview=self._compact_preview(preview),
            )

        if action == "analyze":
            analysis = self.analysis_service.analyze(active_file)
            return self._new_message(
                "assistant",
                self._analysis_message_text(active_file, analysis),
                analysis=self._compact_analysis(analysis),
            )

        if action == "chart":
            try:
                chart_type = self._detect_chart_type(normalized)
                x_column, y_column = self._extract_chart_columns(text)
                chart = self.chart_service.generate_chart(
                    active_file,
                    chart_type,
                    x_column=x_column,
                    y_column=y_column,
                )
            except FileServiceError as exc:
                return self._new_message("assistant", f"Не удалось построить график: {exc}")

            return self._new_message(
                "assistant",
                f"Готово: построил {chart_type}-график для файла «{active_file.original_name}» и сохранил его на сервере.",
                attachments=[self._artifact_chip(chart, "chart")],
            )

        if action == "report":
            try:
                preview = self.file_service.build_preview_context(active_file)
                analysis = self.analysis_service.analyze(active_file)
                chart_records = self.file_service.get_output_artifacts(active_file.file_id)["charts"]
                if not chart_records:
                    self.chart_service.generate_default_charts(active_file)
                    chart_records = self.file_service.get_output_artifacts(active_file.file_id)["charts"]
                report = self.report_service.generate_report(active_file, analysis, chart_records[:3], preview)
            except FileServiceError as exc:
                return self._new_message("assistant", f"Не удалось собрать отчёт: {exc}")

            return self._new_message(
                "assistant",
                f"DOCX-отчёт для файла «{active_file.original_name}» готов. Его можно открыть или скачать.",
                attachments=[self._artifact_chip(report, "report")],
            )

        analysis = self.analysis_service.analyze(active_file)
        charts = self.file_service.get_output_artifacts(active_file.file_id)["charts"]
        export_record = self.export_service.generate_markdown_summary(
            conversation["conversation_id"],
            conversation.get("messages", []),
            stored_file=active_file,
            analysis=analysis,
            charts=charts[:3],
        )
        return self._new_message(
            "assistant",
            "Сохранил краткую сводку в markdown-файл. Туда вошли ключевые выводы, параметры файла и последние сообщения чата.",
            attachments=[self._artifact_chip(export_record, "export")],
        )

    def _new_message(
        self,
        role: str,
        text: str,
        attachments: list[dict[str, Any]] | None = None,
        preview: dict[str, Any] | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at = _utc_now()
        return {
            "message_id": uuid4().hex,
            "role": role,
            "text": text,
            "created_at": created_at,
            "time_label": _time_label(created_at),
            "attachments": attachments or [],
            "preview": preview,
            "analysis": analysis,
        }

    def _persist(self, conversation: dict[str, Any]) -> dict[str, Any]:
        conversation["updated_at"] = _utc_now()
        self._save_conversation(conversation)
        return conversation

    def _save_conversation(self, conversation: dict[str, Any]) -> None:
        self._chat_path(conversation["conversation_id"]).write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _chat_path(self, conversation_id: str) -> Path:
        return self.chat_dir / f"{conversation_id}.json"

    def _get_active_file(self, conversation: dict[str, Any]) -> StoredFile | None:
        file_id = conversation.get("active_file_id")
        if not file_id:
            return None
        try:
            return self.file_service.get_file(file_id)
        except FileServiceError:
            return None

    def _upsert_file_record(self, records: list[dict[str, Any]], stored_file: StoredFile) -> list[dict[str, Any]]:
        filtered = [item for item in records if item.get("file_id") != stored_file.file_id]
        filtered.append(
            {
                "file_id": stored_file.file_id,
                "original_name": stored_file.original_name,
                "kind": stored_file.kind,
                "created_at": stored_file.created_at,
                "time_label": _time_label(stored_file.created_at),
                "size_kb": round(stored_file.size_bytes / 1024, 2),
                "storage_url": f"/storage/{stored_file.relative_path}",
            }
        )
        return filtered

    def _file_chip(self, stored_file: StoredFile) -> dict[str, Any]:
        return {
            "title": "Вложение",
            "kind": "file",
            "name": stored_file.original_name,
            "description": f"{stored_file.kind} · {round(stored_file.size_bytes / 1024, 2)} KB",
        }

    def _stored_file_attachment(self, stored_file: StoredFile) -> dict[str, Any]:
        payload = self._file_chip(stored_file)
        payload.update(
            {
                "download_url": f"/storage/{stored_file.relative_path}",
                "storage_url": f"/storage/{stored_file.relative_path}",
                "preview_url": f"/storage/{stored_file.relative_path}",
            }
        )
        return payload

    def _artifact_chip(self, artifact: dict[str, Any], kind: str) -> dict[str, Any]:
        return {
            "title": artifact.get("title", artifact.get("file_name", "Файл")),
            "kind": kind,
            "name": artifact["file_name"],
            "description": artifact.get("description", ""),
            "storage_url": artifact.get("storage_url"),
            "download_url": artifact.get("download_url"),
            "preview_url": artifact.get("storage_url") if kind == "chart" else None,
        }

    def _compact_preview(self, preview: dict[str, Any]) -> dict[str, Any]:
        if preview["kind"] == "table":
            return {
                "kind": "table",
                "headers": preview["table_headers"][:6],
                "rows": [row[:6] for row in preview["table_rows"][:6]],
                "meta": [
                    f"{preview['row_count']} строк",
                    f"{preview['column_count']} колонок",
                    f"{len(preview['numeric_columns'])} numeric",
                ],
            }
        return {
            "kind": "image",
            "image_url": preview["storage_url"],
            "meta": [
                f"{preview['image_width']}x{preview['image_height']}",
                preview["image_mode"],
                preview["image_format"],
            ],
        }

    def _compact_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": analysis["summary"],
            "insights": analysis.get("insights", [])[:4],
            "stats": analysis.get("stats", [])[:6],
        }

    def _analysis_message_text(self, stored_file: StoredFile, analysis: dict[str, Any]) -> str:
        summary = analysis["summary"]
        return (
            f"Анализ для «{stored_file.original_name}» готов. "
            f"База: {summary['rows']} x {summary['columns']}, "
            f"числовых метрик: {summary['numeric_columns']}, "
            f"пропусков: {summary['missing_cells']}."
        )

    def _detect_action(self, normalized: str) -> str:
        if any(token in normalized for token in ("граф", "chart", "plot", "hist", "bar", "line", "диаграм")):
            return "chart"
        if any(token in normalized for token in ("отч", "report", "docx")):
            return "report"
        if any(token in normalized for token in ("сохран", "экспорт", "markdown", ".md", "txt")):
            return "save"
        if any(token in normalized for token in ("анализ", "проанализ", "статист", "summary", "свод")):
            return "analyze"
        if any(token in normalized for token in ("preview", "предпрос", "покажи", "просмотр")):
            return "preview"
        return "help"

    def _detect_chart_type(self, normalized: str) -> str:
        if any(token in normalized for token in ("hist", "гист")):
            return "histogram"
        if any(token in normalized for token in ("line", "лине")):
            return "line"
        return "bar"

    def _extract_chart_columns(self, text: str) -> tuple[str | None, str | None]:
        x_match = re.search(r"(?:x|x_column|ось x)\s*[:=]\s*[\"'«]?([^,\n;»\"]+)", text, flags=re.IGNORECASE)
        y_match = re.search(r"(?:y|y_column|ось y)\s*[:=]\s*[\"'«]?([^,\n;»\"]+)", text, flags=re.IGNORECASE)
        x_column = x_match.group(1).strip() if x_match else None
        y_column = y_match.group(1).strip() if y_match else None
        return x_column, y_column

    def _collect_saved_artifacts(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        seen: set[str] = set()
        for message in reversed(conversation.get("messages", [])):
            for attachment in message.get("attachments", []):
                download_url = attachment.get("download_url")
                if not download_url or download_url in seen:
                    continue
                if attachment.get("kind") not in {"chart", "report", "export"}:
                    continue
                saved.append(attachment)
                seen.add(download_url)
        return saved
