from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.services.file_service import StoredFile


class ExportService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def generate_markdown_summary(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        stored_file: StoredFile | None = None,
        analysis: dict[str, Any] | None = None,
        charts: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        file_name = self._build_file_name(conversation_id)
        output_path = self.settings.output_dir / file_name

        lines = [
            "# AI Chat Summary",
            "",
            f"- Conversation ID: `{conversation_id}`",
            f"- Generated at: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]

        if stored_file:
            lines.extend(
                [
                    f"- Active file: `{stored_file.original_name}`",
                    f"- File type: `{stored_file.kind}`",
                ]
            )

        if analysis:
            summary = analysis.get("summary", {})
            lines.extend(
                [
                    "",
                    "## Analysis Snapshot",
                    "",
                    f"- Rows / height: {summary.get('rows', 0)}",
                    f"- Columns / width: {summary.get('columns', 0)}",
                    f"- Numeric metrics: {summary.get('numeric_columns', 0)}",
                    f"- Missing cells: {summary.get('missing_cells', 0)}",
                ]
            )
            insights = analysis.get("insights", [])[:5]
            if insights:
                lines.extend(["", "## Key Insights", ""])
                lines.extend([f"- {insight}" for insight in insights])

        if charts:
            lines.extend(["", "## Saved Charts", ""])
            lines.extend([f"- `{chart['file_name']}`" for chart in charts])

        transcript = messages[-8:]
        if transcript:
            lines.extend(["", "## Recent Conversation", ""])
            for message in transcript:
                role = str(message.get("role", "assistant")).capitalize()
                text = str(message.get("text", "")).strip() or "(empty)"
                lines.append(f"### {role}")
                lines.append("")
                lines.append(text)
                lines.append("")

        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return {
            "title": "Markdown summary",
            "kind": "export",
            "file_name": file_name,
            "storage_url": f"/storage/outputs/{file_name}",
            "download_url": f"/download/{file_name}",
            "description": "Markdown-файл с краткой сводкой и последними сообщениями чата.",
        }

    def _build_file_name(self, conversation_id: str) -> str:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
        safe_prefix = Path(conversation_id).name
        return f"{safe_prefix}__summary__{timestamp}.md"
