from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.services.file_service import StoredFile

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    OpenAI = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class AIServiceError(RuntimeError):
    """Base AI service error."""


class AIServiceConfigurationError(AIServiceError):
    """Raised when OpenAI is not configured."""


class AIServiceRequestError(AIServiceError):
    """Raised when the OpenAI API request fails."""


@dataclass
class AIPlan:
    assistant_message: str
    actions: list[dict[str, Any]]


class AIService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.openai_api_key) and OpenAI is not None

    @property
    def model_name(self) -> str:
        return self.settings.openai_model

    def plan_response(
        self,
        *,
        conversation_messages: list[dict[str, Any]],
        user_text: str,
        active_file: StoredFile | None,
        active_file_context: dict[str, Any] | None,
    ) -> AIPlan:
        if OpenAI is None:
            raise AIServiceConfigurationError(
                "Python-пакет `openai` не установлен. Выполните `pip install -r requirements.txt`."
            )
        if not self.settings.openai_api_key:
            raise AIServiceConfigurationError(
                "Не задан `OPENAI_API_KEY` в `.env`."
            )

        client = self._get_client()
        try:
            response = client.responses.create(
                model=self.settings.openai_model,
                instructions=self._system_prompt(),
                input=self._build_input(
                    conversation_messages=conversation_messages,
                    user_text=user_text,
                    active_file=active_file,
                    active_file_context=active_file_context,
                ),
            )
        except Exception as exc:  # pragma: no cover - depends on external API/network
            logger.exception("OpenAI request failed")
            raise AIServiceRequestError(str(exc)) from exc

        payload = self._parse_response_text(getattr(response, "output_text", ""))
        logger.info("OpenAI plan received with %s actions", len(payload.actions))
        return payload

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def _build_input(
        self,
        *,
        conversation_messages: list[dict[str, Any]],
        user_text: str,
        active_file: StoredFile | None,
        active_file_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        history = []
        for message in conversation_messages[-self.settings.openai_max_history_messages :]:
            history.append(
                {
                    "role": message.get("role", "assistant"),
                    "text": message.get("text", ""),
                }
            )

        current_request = user_text.strip() or "Пользователь загрузил файл без дополнительного текста."
        prompt_payload = {
            "current_user_message": current_request,
            "recent_messages": history,
            "active_file": active_file_context,
            "available_actions": [
                {"type": "preview"},
                {"type": "analyze"},
                {
                    "type": "generate_chart",
                    "chart_type": "histogram | bar | line",
                    "x_column": "string | null",
                    "y_column": "string | null",
                },
                {"type": "generate_report"},
                {"type": "save_summary"},
            ],
            "response_contract": {
                "assistant_message": "string",
                "actions": "array",
            },
        }

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Верни только JSON без markdown-обёртки. "
                    "Строго следуй схеме из `response_contract`.\n\n"
                    + json.dumps(prompt_payload, ensure_ascii=False, indent=2)
                ),
            },
        ]

        if active_file and active_file.kind == "image":
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._image_data_url(active_file),
                }
            )

        return [{"role": "user", "content": content}]

    def _system_prompt(self) -> str:
        return """{
  "system_prompt": {
    "metadata": {
      "role": "AI Data Assistant",
      "language": "Russian",
      "domain": "Финансовая аналитика и маркетплейсы"
    },
    "context": "Ты — AI Data Assistant для веб-приложения на русском языке, опытный финансовый аналитик и эксперт по экономике маркетплейсов. Твоя задача — помогать селлерам, менеджерам и финансовым директорам малого бизнеса анализировать их коммерческие показатели. Пользователи не обладают навыками программирования, поэтому твои ответы должны быть понятными, структурными и с фокусом на бизнес-решения. Ты помогаешь анализировать таблицы и изображения.",
    "actions": {
      "description": "Проанализируй загруженные пользователем файлы (CSV, Excel, JSON). Найди ключевые инсайты (лидеры продаж, аномалии в расходах, кассовые разрывы). Для ответа на запросы пользователя используй среду выполнения кода (Python/Pandas), чтобы делать точные расчеты, и строй понятные графики (Matplotlib/Seaborn). Не придумывай действия вне разрешённого списка.",
      "logic": [
        "Если пользователь просит график, отчёт, анализ или сохранение, добавь соответствующее действие в `actions`.",
        "Если действий не требуется, верни пустой список `actions`."
      ]
    },
    "rules": [
      "Используй только те данные, которые есть в загруженных файлах. Категорически запрещено выдумывать цифры, артикулы или статистику (никаких галлюцинаций).",
      "Если данных для ответа недостаточно или файл поврежден, прямо сообщи об этом и скажи, каких колонок не хватает.",
      "Всегда форматируй текстовый ответ с помощью Markdown (используй жирный шрифт для главных цифр и списки для перечислений).",
      "Если пользователь просит график, всегда пиши и выполняй код для его создания.",
      "Не объясняй пользователю сам Python-код, выдавай только готовый бизнес-результат и график.",
      "Если пользователь просит отчет, всегда используй библиотеку python-docx для создания структурированного документа.",
      "При построении графиков используй стиль seaborn и обязательно подписывай оси на русском языке.",
      "Если активный файл — изображение, используй визуальное содержимое изображения в ответе.",
      "Если активный файл — таблица, опирайся на переданные поля, preview, summary и признаки колонок.",
      "Объясняешь выводы простым языком и при необходимости выбираешь действия для приложения.",
      "В конце каждого анализа делай краткий блок 'Рекомендация для бизнеса' из 2-3 пунктов."
    ],
    "example": {
      "user_query": "Какие товары принесли больше всего прибыли в марте по файлу отчета?",
      "assistant_response": "На основе файла sales_report.csv я проанализировал данные за март. \\n\\n**Топ-3 прибыльных товара:**\\n1. **Умная колонка X** — Прибыль: 150 000 руб. (Маржинальность: 35%)\\n2. **Беспроводные наушники Y** — Прибыль: 120 000 руб. (Маржинальность: 40%)\\n3. **Чехол для смартфона Z** — Прибыль: 45 000 руб. (Маржинальность: 60%)\\n\\n⚠️ **Инсайт:** Обратите внимание, что чехлы (Z) приносят меньше абсолютной прибыли, но имеют самую высокую маржинальность. Возможно, стоит увеличить бюджет на их продвижение.\\n\\n**Рекомендация для бизнеса:**\\n— Рассмотреть возможность расширения ассортимента аксессуаров (категория Z).\\n— Проверить остатки по позиции 'Умная колонка X', так как это основной драйвер прибыли.",
      "internal_actions": ["generate_graph", "generate_docx_report"]
    }
  }
}"""

    def _parse_response_text(self, raw_text: str) -> AIPlan:
        text = (raw_text or "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            text = text.removeprefix("json").strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("OpenAI returned non-JSON output; falling back to plain text")
            return AIPlan(assistant_message=text or "Не удалось разобрать ответ модели.", actions=[])

        assistant_message = str(payload.get("assistant_message", "")).strip()
        if not assistant_message:
            assistant_message = "Готово. Я обработал запрос."

        actions: list[dict[str, Any]] = []
        for item in payload.get("actions", []):
            if not isinstance(item, dict):
                continue
            action_type = str(item.get("type", "")).strip()
            if action_type not in {"preview", "analyze", "generate_chart", "generate_report", "save_summary"}:
                continue
            normalized = {"type": action_type}
            if action_type == "generate_chart":
                chart_type = str(item.get("chart_type", "bar")).strip().lower()
                normalized["chart_type"] = chart_type if chart_type in {"histogram", "bar", "line"} else "bar"
                normalized["x_column"] = self._clean_optional_text(item.get("x_column"))
                normalized["y_column"] = self._clean_optional_text(item.get("y_column"))
            actions.append(normalized)

        return AIPlan(assistant_message=assistant_message, actions=actions[:4])

    def _clean_optional_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _image_data_url(self, stored_file: StoredFile) -> str:
        mime_type = stored_file.content_type or "image/png"
        encoded = base64.b64encode(stored_file.path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
