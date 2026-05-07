from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.routes.pages import render_page
from app.services.file_service import (
    EmptyFileError,
    FileReadError,
    FileService,
    FileTooLargeError,
    UnsupportedFileError,
)


settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))
file_service = FileService(settings)

router = APIRouter()


@router.post("/upload")
async def upload_file(request: Request, data_file: UploadFile = File(...)):
    try:
        stored_file = await file_service.save_upload(data_file)
        preview_context = file_service.build_preview_context(stored_file)
        artifacts = file_service.get_output_artifacts(stored_file.file_id)
    except (UnsupportedFileError, FileTooLargeError, EmptyFileError, FileReadError) as exc:
        response = render_page(
            request,
            "index.html",
            "partials/index_content.html",
            {
                "request": request,
                "page_title": "Data Assistant",
                "error_message": str(exc),
                "success_message": None,
            },
        )
        response.status_code = status.HTTP_400_BAD_REQUEST
        return response

    if request.headers.get("HX-Request") != "true":
        return RedirectResponse(url=f"/preview/{stored_file.file_id}", status_code=status.HTTP_303_SEE_OTHER)

    response = templates.TemplateResponse(
        "partials/preview_content.html",
        {
            "request": request,
            "page_title": f"Preview | {stored_file.original_name}",
            "preview": preview_context,
            "artifacts": artifacts,
            "analysis": None,
            "error_message": None,
            "success_message": "Файл загружен. Preview подготовлен без перезагрузки страницы.",
        },
    )
    response.headers["HX-Push-Url"] = f"/preview/{stored_file.file_id}"
    return response
