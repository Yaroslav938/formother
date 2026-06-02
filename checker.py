import base64
import json
import re
from io import BytesIO

import anthropic
from pypdf import PdfReader
from docx import Document
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

# --- Конфигурация ---
MAX_IMAGE_SIDE = 1024
JPEG_QUALITY = 85
MODEL = "claude-opus-4-5"
BASE_URL = "https://api.east-api-3.org"
MAX_TOKENS = 3000
MIN_TEXT_LEN = 15


# ============================================================
#  OCR — EasyOCR (чистый pip, работает на Streamlit Cloud)
# ============================================================
_OCR_READER = None


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            _OCR_READER = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
        except Exception:
            _OCR_READER = False
    return _OCR_READER


def _ocr_preprocess(img: Image.Image) -> Image.Image:
    """Улучшает контраст и резкость для лучшего распознавания."""
    try:
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(1.8)
        img = img.filter(ImageFilter.SHARPEN)
        return img
    except Exception:
        return img


def run_local_ocr(img: Image.Image) -> str:
    """Распознаёт текст с изображения через EasyOCR. Возвращает пустую строку при неудаче."""
    reader = _get_ocr_reader()
    if not reader:
        return ""

    try:
        img_rgb = np.array(img.convert("RGB"))
        result = reader.readtext(img_rgb)
        if result:
            lines = []
            for detection in result:
                text = detection[1].strip()
                if text:
                    lines.append(text)
            return "\n".join(lines)
    except Exception:
        pass
    return ""


def ocr_available() -> bool:
    return _get_ocr_reader() is not None


# ============================================================
#  Подготовка изображения
# ============================================================
def _prep_image(img: Image.Image) -> Image.Image:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_SIDE:
        ratio = MAX_IMAGE_SIDE / max(img.size)
        img = img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
            Image.LANCZOS,
        )
    return img


def _img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
#  Извлечение контента из файла
# ============================================================
def extract_content(file) -> dict:
    ext = file.name.rsplit(".", 1)[-1].lower()

    try:
        file.seek(0)
    except Exception:
        pass
    raw = file.read()

    # ---------- PDF ----------
    if ext == "pdf":
        text = ""
        try:
            reader = PdfReader(BytesIO(raw))
            text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        except Exception:
            text = ""

        if len(text) >= MIN_TEXT_LEN:
            return {"type": "text", "data": text}

        # PDF без текста — пробуем отрендерить через PyMuPDF
        img = _render_pdf_first_page(raw)
        if img is not None:
            img = _prep_image(img)
            ocr = run_local_ocr(img)
            return {
                "type": "image",
                "data": _img_to_b64(img),
                "media_type": "image/jpeg",
                "ocr": ocr,
            }
        return {"type": "text", "data": text or "[PDF не содержит текста]", "ocr": ""}

    # ---------- DOCX ----------
    if ext == "docx":
        try:
            doc = Document(BytesIO(raw))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for r in table.rows:
                    parts.extend(c.text for c in r.cells)
            text = "\n".join(t for t in parts if t).strip()
        except Exception:
            text = ""
        return {"type": "text", "data": text or "[DOCX не читается]", "ocr": ""}

    # ---------- Изображения ----------
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    img = Image.open(BytesIO(raw))
    img = _prep_image(img)
    ocr = run_local_ocr(img)
    return {
        "type": "image",
        "data": _img_to_b64(img),
        "media_type": mime,
        "ocr": ocr,
    }


def _render_pdf_first_page(raw: bytes):
    try:
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=200)
        return Image.open(BytesIO(pix.tobytes("png")))
    except Exception:
        pass
    return None


# ============================================================
#  Конфигурация предметов
# ============================================================
SUBJECT_CONFIG = {
    "Английский язык": {
        "role": "опытный учитель английского языка",
        "scores": [
            ("GRAMMAR_SCORE", "Грамматика"),
            ("VOCABULARY_SCORE", "Лексика"),
            ("SPELLING_SCORE", "Орфография"),
            ("COHERENCE_SCORE", "Связность"),
        ],
        "errors": [
            ("GRAMMAR_ERRORS", "🔴 Грамматические ошибки"),
            ("SPELLING_ERRORS", "🟠 Орфографические ошибки"),
            ("VOCABULARY_ERRORS", "🟡 Лексические ошибки"),
        ],
        "level_hint": "уровень владения языком: A1/A2/B1/B2/C1/C2",
    },
    "Русский язык": {
        "role": "опытный учитель русского языка и литературы",
        "scores": [
            ("SPELLING_SCORE", "Орфография"),
            ("PUNCTUATION_SCORE", "Пунктуация"),
            ("GRAMMAR_SCORE", "Грамматика"),
            ("STYLE_SCORE", "Стиль и связность"),
        ],
        "errors": [
            ("SPELLING_ERRORS", "🔴 Орфографические ошибки"),
            ("PUNCTUATION_ERRORS", "🟠 Пунктуационные ошибки"),
            ("STYLE_ERRORS", "🟡 Стилистические/речевые ошибки"),
        ],
        "level_hint": "уровень грамотности: Начальный/Базовый/Уверенный/Высокий",
    },
    "Биология": {
        "role": "опытный учитель биологии",
        "scores": [
            ("KNOWLEDGE_SCORE", "Знание"),
            ("CORRECTNESS_SCORE", "Термины"),
            ("LOGIC_SCORE", "Логика"),
            ("EXAMPLES_SCORE", "Примеры"),
        ],
        "errors": [
            ("FACTUAL_ERRORS", "🔴 Фактические ошибки"),
            ("TERM_ERRORS", "🟠 Ошибки в терминах"),
            ("STRUCTURE_ERRORS", "🟡 Ошибки в логике"),
        ],
        "level_hint": "уровень знаний: Начальный/Базовый/Продвинутый/Эксперт",
    },
    "Математика": {
        "role": "опытный учитель математики",
        "scores": [
            ("CALCULATION_SCORE", "Вычисления"),
            ("LOGIC_SCORE", "Логика"),
            ("METHOD_SCORE", "Метод"),
            ("PRESENTATION_SCORE", "Оформление"),
        ],
        "errors": [
            ("CALCULATION_ERRORS", "🔴 Вычислительные ошибки"),
            ("LOGIC_ERRORS", "🟠 Логические ошибки"),
            ("METHOD_ERRORS", "🟡 Ошибки в методе"),
        ],
        "level_hint": "уровень: Начальный/Базовый/Продвинутый/Эксперт",
    },
    "Физика": {
        "role": "опытный учитель физики",
        "scores": [
            ("CONCEPT_SCORE", "Понимание законов"),
            ("CALCULATION_SCORE", "Вычисления"),
            ("UNITS_SCORE", "Единицы измерения"),
            ("PRESENTATION_SCORE", "Оформление"),
        ],
        "errors": [
            ("CONCEPT_ERRORS", "🔴 Ошибки в законах/понятиях"),
            ("CALCULATION_ERRORS", "🟠 Вычислительные ошибки"),
            ("UNITS_ERRORS", "🟡 Ошибки в единицах измерения"),
        ],
        "level_hint": "уровень: Начальный/Базовый/Продвинутый/Эксперт",
    },
    "Химия": {
        "role": "опытный учитель химии",
        "scores": [
            ("EQUATION_SCORE", "Уравнения реакций"),
            ("CALCULATION_SCORE", "Расчёты"),
            ("TERM_SCORE", "Термины и понятия"),
            ("PRESENTATION_SCORE", "Оформление"),
        ],
        "errors": [
            ("EQUATION_ERRORS", "🔴 Ошибки в уравнениях реакций"),
            ("CALCULATION_ERRORS", "🟠 Расчётные ошибки"),
            ("TERM_ERRORS", "🟡 Ошибки в терминах/формулах"),
        ],
        "level_hint": "уровень: Начальный/Базовый/Продвинутый/Эксперт",
    },
}


def get_subjects():
    return list(SUBJECT_CONFIG.keys())


def get_subject_config(subject: str):
    return SUBJECT_CONFIG.get(subject, SUBJECT_CONFIG["Английский язык"])


# ============================================================
#  Построение промпта
# ============================================================
def build_prompt(subject: str) -> str:
    cfg = get_subject_config(subject)
    score_keys = [k for k, _ in cfg["scores"]]
    error_keys = [k for k, _ in cfg["errors"]]

    score_lines = "\n".join(f'  "{k}": <целое число от 1 до 10>,' for k in score_keys)
    error_lines = "\n".join(
        f'  "{k}": "<краткий список ошибок через точку с запятой, или \\"нет\\">",'
        for k in error_keys
    )

    return f"""Ты — {cfg['role']}. Внимательно проверь домашнюю работу ученика и дай детальный, объективный анализ на русском языке.

Ответь СТРОГО валидным JSON-объектом без markdown, без пояснений до или после. Структура:
{{
  "GRADE": <итоговая оценка, целое число от 1 до 10>,
{score_lines}
{error_lines}
  "STRENGTHS": "<что сделано хорошо, через точку с запятой>",
  "RECOMMENDATIONS": "<конкретные рекомендации по улучшению, через точку с запятой>",
  "LEVEL": "<{cfg['level_hint']}>",
  "SUMMARY": "<краткое общее заключение в 2-3 предложениях>"
}}

Важно: значения ошибок — строки. Если ошибок нет — ставь "нет". Верни ТОЛЬКО JSON."""


# ============================================================
#  Парсинг ответа
# ============================================================
def _extract_json_block(text: str):
    # Убираем markdown
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    # Ищем { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


def _fallback_parse(text: str, subject: str) -> dict:
    cfg = get_subject_config(subject)
    known = {"GRADE", "STRENGTHS", "RECOMMENDATIONS", "LEVEL", "SUMMARY"}
    known |= {k for k, _ in cfg["scores"]}
    known |= {k for k, _ in cfg["errors"]}

    def clean(s):
        s = s.strip()
        s = re.sub(r"[*_`#>]+", "", s)
        return s.strip('"').strip()

    result = {}
    current_key = None
    for line in text.splitlines():
        line = line.strip().lstrip("*#-• ").strip()
        m = re.match(r"^[*_`]*([A-Z_]+)[*_`]*\s*[:：]\s*(.*)$", line)
        if m and m.group(1) in known:
            current_key = m.group(1)
            result[current_key] = clean(m.group(2))
        elif current_key and line:
            result[current_key] = (result[current_key] + " " + clean(line)).strip()
    return result


def _coerce_int(val, default=0) -> int:
    try:
        m = re.search(r"-?\d+", str(val))
        return int(m.group()) if m else default
    except Exception:
        return default


def parse_response(text: str, subject: str) -> dict:
    cfg = get_subject_config(subject)
    data = _extract_json_block(text)
    if data is None or not isinstance(data, dict):
        data = _fallback_parse(text, subject)

    if not data:
        return {"_raw": text, "_parse_failed": True}

    result = {}
    result["GRADE"] = _coerce_int(data.get("GRADE"), 0)
    for k, _ in cfg["scores"]:
        result[k] = _coerce_int(data.get(k), 0)
    for k, _ in cfg["errors"]:
        result[k] = str(data.get(k, "нет")).strip() or "нет"
    for k in ("STRENGTHS", "RECOMMENDATIONS", "LEVEL", "SUMMARY"):
        result[k] = str(data.get(k, "—")).strip() or "—"
    return result


def _collect_text(resp) -> str:
    raw_text = ""
    try:
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                raw_text += block.text
        if not raw_text and resp.content:
            raw_text = getattr(resp.content[0], "text", "") or ""
    except Exception:
        raw_text = str(getattr(resp, "content", ""))
    return raw_text


# ============================================================
#  Диагностика vision
# ============================================================
def diagnose_vision(api_key: str) -> dict:
    from PIL import Image as PILImage, ImageDraw
    secret = "TURTLE7"
    img = PILImage.new("RGB", (400, 160), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 60), secret, fill="black")
    b64 = _img_to_b64(img)

    client = anthropic.Anthropic(api_key=api_key, base_url=BASE_URL)
    out = {"secret": secret, "model": MODEL, "base_url": BASE_URL}
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Какое слово написано на картинке? Ответь только этим словом."},
            ]}],
        )
        answer = _collect_text(resp).strip()
        out["model_in_response"] = getattr(resp, "model", "?")
        out["answer"] = answer
        out["vision_ok"] = secret.lower() in answer.lower()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["vision_ok"] = False
    return out


# ============================================================
#  Проверка работы
# ============================================================
def check_homework(content: dict, api_key: str, subject: str = "Английский язык",
                   debug: bool = False) -> dict:
    prompt = build_prompt(subject)
    client = anthropic.Anthropic(api_key=api_key, base_url=BASE_URL)
    ocr_text = (content.get("ocr") or "").strip()
    dbg = {}

    if content["type"] == "text":
        # PDF/DOCX — простой текст
        body = prompt + "\n\nРабота ученика:\n" + content["data"]
        msg_content = [{"type": "text", "text": body}]
        dbg["режим"] = "текст (PDF/DOCX)"

    elif ocr_text:
        # Изображение с OCR — ОТПРАВЛЯЕМ ТОЛЬКО ТЕКСТ (прокси не поддерживает vision)
        body = prompt + "\n\nТекст работы ученика (распознан с фото):\n" + ocr_text
        msg_content = [{"type": "text", "text": body}]
        dbg["режим"] = "ocr-текст"
        dbg["ocr_длина"] = len(ocr_text)
        dbg["ocr_начало"] = ocr_text[:200]

    else:
        # Изображение без OCR — попытка отправить картинку напрямую
        body = prompt + "\n\nВнимательно рассмотри изображение и проверь работу."
        msg_content = [
            {"type": "image", "source": {"type": "base64",
                 "media_type": content["media_type"], "data": content["data"]}},
            {"type": "text", "text": body},
        ]
        dbg["режим"] = "изображение (без OCR)"
        dbg["ocr_длина"] = 0

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": msg_content}],
        )
        raw_text = _collect_text(resp)
        dbg["ответ_длина"] = len(raw_text)
        dbg["ответ_начало"] = raw_text[:300]
    except Exception as e:
        raw_text = ""
        dbg["ошибка"] = f"{type(e).__name__}: {e}"

    result = parse_response(raw_text, subject)
    result["Предмет"] = subject
    if debug:
        result["_debug"] = dbg
    return result
