from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.services.ai_service import AIService
from app.services.analysis_service import AnalysisService
from app.services.chart_service import ChartService
from app.services.chat_service import ChatNotFoundError, ChatService
from app.services.export_service import ExportService
from app.services.file_service import FileService
from app.services.report_service import ReportService


settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))

file_service = FileService(settings)
analysis_service = AnalysisService(file_service)
chart_service = ChartService(file_service, settings)
report_service = ReportService(file_service, settings)
export_service = ExportService(settings)
ai_service = AIService(settings)
chat_service = ChatService(
    file_service=file_service,
    analysis_service=analysis_service,
    chart_service=chart_service,
    report_service=report_service,
    export_service=export_service,
    ai_service=ai_service,
    settings=settings,
)

router = APIRouter()


def render_page(request: Request, template_name: str, partial_name: str, context: dict[str, Any]):
    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(partial_name, context)
    return templates.TemplateResponse(template_name, context)


@router.get("/")
async def index(request: Request):
    conversation = chat_service.create_conversation()
    return RedirectResponse(url=f"/chat/{conversation['conversation_id']}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/chat/{conversation_id}")
async def chat_page(request: Request, conversation_id: str):
    try:
        conversation = chat_service.get_conversation(conversation_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.") from exc

    return render_page(
        request,
        "chat.html",
        "partials/chat_shell.html",
        chat_service.build_page_context(request, conversation),
    )


@router.get("/preview/{file_id}")
async def preview_page(request: Request, file_id: str):
    stored_file = file_service.get_file(file_id)
    preview_context = file_service.build_preview_context(stored_file)
    artifacts = file_service.get_output_artifacts(file_id)
    return render_page(
        request,
        "preview.html",
        "partials/preview_content.html",
        {
            "request": request,
            "page_title": f"Preview | {stored_file.original_name}",
            "preview": preview_context,
            "artifacts": artifacts,
            "analysis": None,
            "error_message": None,
            "success_message": "Файл загружен. Можно запускать анализ и строить графики.",
        },
    )


@router.get("/results/{file_id}")
async def result_page(request: Request, file_id: str):
    stored_file = file_service.get_file(file_id)
    preview_context = file_service.build_preview_context(stored_file)
    analysis = analysis_service.analyze(stored_file)
    artifacts = file_service.get_output_artifacts(file_id)
    if not artifacts["charts"]:
        chart_service.generate_default_charts(stored_file)
        artifacts = file_service.get_output_artifacts(file_id)

    return render_page(
        request,
        "result.html",
        "partials/result_page_content.html",
        {
            "request": request,
            "page_title": f"Results | {stored_file.original_name}",
            "preview": preview_context,
            "analysis": analysis,
            "artifacts": artifacts,
            "error_message": None,
            "success_message": "Собран актуальный аналитический срез по файлу.",
        },
    )
