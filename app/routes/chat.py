from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.routes.pages import chat_service, render_page
from app.services.chat_service import ChatNotFoundError

router = APIRouter()


@router.post("/chat/{conversation_id}/message")
async def send_message(
    request: Request,
    conversation_id: str,
    message_text: str = Form(default=""),
    data_file: UploadFile | None = File(default=None),
):
    try:
        conversation = await chat_service.process_turn(conversation_id, message_text, data_file)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.") from exc

    return render_page(
        request,
        "chat.html",
        "partials/chat_shell.html",
        chat_service.build_page_context(request, conversation),
    )


@router.post("/chat/{conversation_id}/activate/{file_id}")
async def activate_file(request: Request, conversation_id: str, file_id: str):
    try:
        conversation = chat_service.activate_file(conversation_id, file_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден.") from exc

    return render_page(
        request,
        "chat.html",
        "partials/chat_shell.html",
        chat_service.build_page_context(request, conversation),
    )
