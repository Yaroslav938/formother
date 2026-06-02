"""
checker.py — извлечение контента из файлов и проверка работ через Claude API.

Ключевые улучшения по сравнению с прежней версией:
- Надёжный парсинг ответа модели через JSON (а не построчный «ключ: значение»).
- Запрос к модели в формате structured JSON + защитный fallback-парсер.
- Высокое качество изображений для vision Claude (рукописный текст читается лучше).
- Локальный OCR (tesseract/easyocr) добавляется как дополнительный контекст.
- Защита от пустого текста из сканов-PDF: при пустом тексте PDF рендерится в изображение.
- 6 предметов: Английский, Русский, Биология, Математика, Физика, Химия.
"""

import base64
import json
import re
from io import BytesIO

import anthropic
from pypdf import PdfReader
from docx import Document
from PIL import Image, ImageOps

# --- Конфигурация ---
# Для vision-моделей Anthropic оптимальна сторона ~1568px по длинной стороне.
# Это даёт хорошую читаемость рукописного текста без перерасхода токенов.
MAX_IMAGE_SIDE = 1568
JPEG_QUALITY = 90
MODEL = "claude-opus-4-5"
BASE_URL = "https://api.east-api-3.org"
MAX_TOKENS = 3000

# Минимальная длина текста, ниже которой PDF/DOCX считаем «пустым» (скан/картинка)
MIN_TEXT_LEN = 15


# ============================================================
#  OCR (опционально — как дополнительный контекст для модели)
# ============================================================
# Глобальный кэш RapidOCR (тяжёлая инициализация — делаем один раз)
_RAPID_ENGINE = None


def _get_rapidocr():
    global _RAPID_ENGINE
    if _RAPID_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID_ENGINE = RapidOCR()
    return _RAPID_ENGINE


def ocr_available() -> bool:
    """Есть ли хотя бы один рабочий OCR-движок."""
    # 1) RapidOCR — чистый pip, не требует системных пакетов (приоритет)
    try:
        import rapidocr_onnxruntime  # noqa
        return True
    except Exception:
        pass
    # 2) tesseract — требует системный пакет (на Streamlit Cloud часто не ставится)
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass
    # 3) easyocr
    try:
        import easyocr  # noqa
        return True
    except Exception:
        return False


def _ocr_preprocess(img: Image.Image) -> Image.Image:
    """Повышает качество распознавания: серый + контраст."""
    try:
        from PIL import ImageEnhance, ImageFilter
        g = img.convert("L")
        g = ImageEnhance.Contrast(g).enhance(1.8)
        g = g.filter(ImageFilter.SHARPEN)
        return g
    except Exception:
        return img


def run_local_ocr(img: Image.Image) -> str:
    """Пытается распознать текст локально. Возвращает '' при неудаче."""
    # 1) RapidOCR — работает без системных пакетов (основной вариант для Streamlit Cloud)
    try:
        import numpy as np
        engine = _get_rapidocr()
        result, _ = engine(np.array(img.convert("RGB")))
        if result:
            text = "\n".join(item[1] for item in result)
            if text and text.strip():
                return text.strip()
    except Exception:
        pass

    prepped = _ocr_preprocess(img)

    # 2) pytesseract (rus+eng, с fallback на eng)
    try:
        import pytesseract
        for langs in ("rus+eng", "eng", "rus"):
            try:
                text = pytesseract.image_to_string(prepped, lang=langs)
            except Exception:
                continue
            if text and text.strip():
                return text.strip()
    except Exception:
        pass

    # 3) easyocr
    try:
        import easyocr
        import numpy as np
        reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
        result = reader.readtext(np.array(img))
        text = "\n".join(item[1] for item in result)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    return ""


# ============================================================
#  Подготовка изображения
# ============================================================
def _prep_image(img: Image.Image) -> Image.Image:
    """Корректирует ориентацию по EXIF и масштабирует под vision."""
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
#  Извлечение контента из загруженного файла
# ============================================================
def extract_content(file) -> dict:
    """
    Возвращает словарь одного из видов:
      {"type": "text", "data": <str>, "ocr": <str|''>}
      {"type": "image", "data": <b64>, "media_type": <str>, "ocr": <str|''>}

    Гарантированно сбрасывает указатель файла перед чтением,
    чтобы повторное чтение (превью + обработка) не давало пустоту.
    """
    ext = file.name.rsplit(".", 1)[-1].lower()

    # Всегда читаем сырые байты один раз — это устойчиво к положению указателя.
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
            return {"type": "text", "data": text, "ocr": ""}

        # PDF без текстового слоя (скан) — пробуем отрендерить первую страницу в картинку
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
        # Если рендер недоступен — возвращаем то, что есть (пусть и мало)
        return {"type": "text", "data": text or "[PDF не содержит распознаваемого текста]", "ocr": ""}

    # ---------- DOCX ----------
    if ext == "docx":
        try:
            doc = Document(BytesIO(raw))
            parts = [p.text for p in doc.paragraphs]
            # Текст из таблиц тоже учитываем
            for table in doc.tables:
                for r in table.rows:
                    parts.extend(c.text for c in r.cells)
            text = "\n".join(t for t in parts if t).strip()
        except Exception:
            text = ""
        return {"type": "text", "data": text or "[DOCX пустой или не читается]", "ocr": ""}

    # ---------- Изображения ----------
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    img = Image.open(BytesIO(raw))
    img = _prep_image(img)
    ocr = run_local_ocr(img)
    return {
        "type": "image",
        "data": _img_to_b64(img),
        "media_type": "image/jpeg",  # сохраняем как JPEG после обработки
        "ocr": ocr,
    }


def _render_pdf_first_page(raw: bytes):
    """Пытается отрендерить первую страницу PDF в PIL.Image (если есть pdf2image/pymupdf)."""
    # 1) PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=raw, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=200)
        return Image.open(BytesIO(pix.tobytes("png")))
    except Exception:
        pass
    # 2) pdf2image (требует poppler)
    try:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(raw, dpi=200, first_page=1, last_page=1)
        if pages:
            return pages[0]
    except Exception:
        pass
    return None


# ============================================================
#  Конфигурация предметов (поля и критерии)
# ============================================================
# Для каждого предмета задаём 4 критерия оценивания (ключи и заголовки),
# а также секции ошибок. Это используется и в промпте, и в UI.
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


def get_subjects() -> list:
    return list(SUBJECT_CONFIG.keys())


def get_subject_config(subject: str) -> dict:
    return SUBJECT_CONFIG.get(subject, SUBJECT_CONFIG["Английский язык"])


# ============================================================
#  Построение промпта (просим строго JSON)
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

Если работа представлена изображением — сначала аккуратно распознай весь текст (включая рукописный), затем проверяй.

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

Важно: значения ошибок — это строки. Если ошибок нет — поставь "нет". Не добавляй никаких полей, кроме перечисленных. Верни ТОЛЬКО JSON."""


# ============================================================
#  Парсинг ответа модели
# ============================================================
def _extract_json_block(text: str):
    """Извлекает первый JSON-объект из текста (на случай обёрток/markdown)."""
    # Убираем markdown-ограждения ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Берём подстроку от первой { до последней }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def _fallback_line_parse(text: str, subject: str) -> dict:
    """Запасной построчный парсер «КЛЮЧ: значение» (многострочно-устойчивый)."""
    cfg = get_subject_config(subject)
    known = {"GRADE", "STRENGTHS", "RECOMMENDATIONS", "LEVEL", "SUMMARY"}
    known |= {k for k, _ in cfg["scores"]}
    known |= {k for k, _ in cfg["errors"]}

    def clean(s: str) -> str:
        # Убираем markdown-разметку (** __ ` # > -) и кавычки по краям
        s = s.strip()
        s = re.sub(r"[*_`#>]+", "", s)
        return s.strip().strip('"').strip()

    result = {}
    current_key = None
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("*#-• ").strip()
        # Ключ может быть обёрнут в markdown: **GRADE:** или __GRADE__:
        m = re.match(r"^[*_`]*([A-Z_]+)[*_`]*\s*[:：]\s*(.*)$", line)
        if m and m.group(1) in known:
            current_key = m.group(1)
            result[current_key] = clean(m.group(2))
        elif current_key and line:
            # продолжение многострочного значения
            result[current_key] = (result[current_key] + " " + clean(line)).strip()
    return result


def _coerce_int(val, default=0) -> int:
    try:
        m = re.search(r"-?\d+", str(val))
        return int(m.group()) if m else default
    except Exception:
        return default


def parse_response(text: str, subject: str) -> dict:
    """Возвращает нормализованный словарь результата проверки."""
    cfg = get_subject_config(subject)
    data = _extract_json_block(text)
    if data is None or not isinstance(data, dict):
        data = _fallback_line_parse(text, subject)

    # Если парсинг полностью провалился — возвращаем сигнал об ошибке
    if not data:
        return {"_raw": text, "_parse_failed": True}

    result = {}
    # Числовые оценки
    result["GRADE"] = _coerce_int(data.get("GRADE"), 0)
    for k, _ in cfg["scores"]:
        result[k] = _coerce_int(data.get(k), 0)
    # Текстовые поля
    for k, _ in cfg["errors"]:
        result[k] = str(data.get(k, "нет")).strip() or "нет"
    for k in ("STRENGTHS", "RECOMMENDATIONS", "LEVEL", "SUMMARY"):
        result[k] = str(data.get(k, "—")).strip() or "—"
    return result


# ============================================================
#  Основная проверка
# ============================================================
def _collect_text(resp) -> str:
    """Собирает весь текст из ответа (устойчиво к формату прокси)."""
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


def check_homework(content: dict, api_key: str, subject: str = "Английский язык",
                   debug: bool = False) -> dict:
    prompt = build_prompt(subject)
    client = anthropic.Anthropic(api_key=api_key, base_url=BASE_URL)

    ocr_text = (content.get("ocr") or "").strip()
    dbg = {}

    if content["type"] == "text":
        body = prompt + "\n\nРабота ученика:\n" + content["data"]
        msg_content = [{"type": "text", "text": body}]
        dbg["режим"] = "текст"
    else:
        # ВАЖНО: распознанный текст передаём КАК РАБОТУ УЧЕНИКА.
        # Это страховка на случай, если прокси НЕ передаёт изображение в модель
        # (тогда раньше приходило «работа не предоставлена»).
        text_block = prompt
        if ocr_text:
            text_block += (
                "\n\nНиже — текст работы ученика, распознанный с изображения "
                "(может содержать редкие ошибки распознавания). Если видишь изображение — "
                "опирайся на него; иначе проверяй по этому тексту:\n\n" + ocr_text
            )
        msg_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content["media_type"],
                    "data": content["data"],
                },
            },
            {"type": "text", "text": text_block},
        ]
        dbg["режим"] = "изображение"
        dbg["длина_base64"] = len(content.get("data", ""))
        dbg["длина_ocr"] = len(ocr_text)

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": msg_content}],
    )
    raw_text = _collect_text(resp)
    dbg["модель_в_ответе"] = getattr(resp, "model", "?")
    dbg["длина_ответа"] = len(raw_text)
    dbg["ответ_начало"] = raw_text[:400]

    result = parse_response(raw_text, subject)
    result["Предмет"] = subject
    if debug:
        result["_debug"] = dbg
    return result


def diagnose_vision(api_key: str) -> dict:
    """
    Проверяет, поддерживает ли прокси/модель работу с изображениями.
    Генерирует картинку с чётким словом и просит модель его прочитать.
    Если модель не называет слово — значит изображения до неё не доходят (прокси без vision).
    """
    from PIL import Image, ImageDraw
    secret = "TURTLE7"
    img = Image.new("RGB", (400, 160), "white")
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
                {"type": "text", "text": "Какое слово/код написано на картинке? Ответь только этим словом."},
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
