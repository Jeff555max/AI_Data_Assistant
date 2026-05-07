from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.services.analysis_service import AnalysisService
from app.services.chart_service import ChartService
from app.services.file_service import FileReadError, FileService
from app.services.report_service import ReportService


settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))

file_service = FileService(settings)
analysis_service = AnalysisService(file_service)
chart_service = ChartService(file_service, settings)
report_service = ReportService(file_service, settings)

router = APIRouter()


def _panel_context(
    request: Request,
    file_id: str,
    success_message: str | None = None,
    error_message: str | None = None,
):
    stored_file = file_service.get_file(file_id)
    preview_context = file_service.build_preview_context(stored_file)
    analysis = analysis_service.analyze(stored_file)
    artifacts = file_service.get_output_artifacts(file_id)
    return {
        "request": request,
        "preview": preview_context,
        "analysis": analysis,
        "artifacts": artifacts,
        "success_message": success_message,
        "error_message": error_message,
    }


@router.post("/actions/analyze/{file_id}")
async def analyze_action(request: Request, file_id: str):
    context = _panel_context(request, file_id, success_message="Анализ обновлен.")
    return templates.TemplateResponse("partials/result_panel.html", context)


@router.post("/actions/chart/{file_id}")
async def chart_action(
    request: Request,
    file_id: str,
    chart_type: str = Form(...),
    x_column: str | None = Form(default=None),
    y_column: str | None = Form(default=None),
):
    try:
        stored_file = file_service.get_file(file_id)
        chart_service.generate_chart(stored_file, chart_type, x_column=x_column or None, y_column=y_column or None)
        context = _panel_context(request, file_id, success_message="График построен и сохранен.")
    except FileReadError as exc:
        context = _panel_context(request, file_id, error_message=str(exc))
    return templates.TemplateResponse("partials/result_panel.html", context)


@router.post("/actions/report/{file_id}")
async def report_action(request: Request, file_id: str):
    stored_file = file_service.get_file(file_id)
    preview_context = file_service.build_preview_context(stored_file)
    analysis = analysis_service.analyze(stored_file)
    artifacts = file_service.get_output_artifacts(file_id)

    chart_records = artifacts["charts"]
    if not chart_records:
        chart_records = chart_service.generate_default_charts(stored_file)
        artifacts = file_service.get_output_artifacts(file_id)

    report_service.generate_report(stored_file, analysis, chart_records[:3], preview_context)
    context = _panel_context(request, file_id, success_message="DOCX-отчет сформирован и готов к скачиванию.")
    return templates.TemplateResponse("partials/result_panel.html", context)


@router.get("/download/{artifact_name}")
async def download_artifact(artifact_name: str):
    candidate = settings.output_dir / artifact_name
    if not candidate.exists() or candidate.parent != settings.output_dir:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Файл не найден.")
    return FileResponse(path=candidate, filename=candidate.name)
