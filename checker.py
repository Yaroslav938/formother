import base64
import anthropic
import platform
import pytesseract
from pypdf import PdfReader
from docx import Document
from io import BytesIO
from PIL import Image

# Настройка tesseract для Linux (Streamlit Cloud)
if platform.system() == "Linux":
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

MAX_PIXELS = 1024


def extract_content(file):
    ext = file.name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        text = "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
        return {"type": "text", "data": text}
    elif ext == "docx":
        text = "\n".join(p.text for p in Document(file).paragraphs)
        return {"type": "text", "data": text}
    else:
        raw = file.read()
        img = Image.open(BytesIO(raw))
        if max(img.size) > MAX_PIXELS:
            ratio = MAX_PIXELS / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        text = pytesseract.image_to_string(img, lang="eng+rus")
        return {"type": "text", "data": text}


def parse_response(text):
    result = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


SUBJECTS = {
    "Английский язык": """Ты — опытный учитель английского языка. Проверь домашнюю работу ученика и дай детальный анализ.

Ответь СТРОГО в следующем формате (каждое поле с новой строки):
GRADE: <число от 1 до 10>
GRAMMAR_SCORE: <число от 1 до 10>
VOCABULARY_SCORE: <число от 1 до 10>
SPELLING_SCORE: <число от 1 до 10>
COHERENCE_SCORE: <число от 1 до 10>
GRAMMAR_ERRORS: <список грамматических ошибок через точку с запятой, или "нет">
SPELLING_ERRORS: <список орфографических ошибок через точку с запятой, или "нет">
VOCABULARY_ERRORS: <список лексических ошибок через точку с запятой, или "нет">
STRENGTHS: <что сделано хорошо, через точку с запятой>
RECOMMENDATIONS: <конкретные рекомендации по улучшению, через точку с запятой>
LEVEL: <уровень ученика: A1/A2/B1/B2/C1/C2>
SUMMARY: <краткое общее заключение в 2-3 предложениях>""",

    "Биология": """Ты — опытный учитель биологии. Проверь домашнюю работу ученика и дай детальный анализ.

Ответь СТРОГО в следующем формате (каждое поле с новой строки):
GRADE: <число от 1 до 10>
KNOWLEDGE_SCORE: <оценка знания материала от 1 до 10>
CORRECTNESS_SCORE: <оценка правильности терминов и фактов от 1 до 10>
LOGIC_SCORE: <оценка логичности изложения от 1 до 10>
EXAMPLES_SCORE: <оценка использования примеров от 1 до 10>
FACTUAL_ERRORS: <список фактических ошибок через точку с запятой, или "нет">
TERM_ERRORS: <список ошибок в использовании терминов через точку с запятой, или "нет">
STRUCTURE_ERRORS: <список ошибок в структуре и логике ответа через точку с запятой, или "нет">
STRENGTHS: <что сделано хорошо, через точку с запятой>
RECOMMENDATIONS: <конкретные рекомендации по улучшению, через точку с запятой>
LEVEL: <уровень знаний: Начальный/Базовый/Продвинутый/Эксперт>
SUMMARY: <краткое общее заключение в 2-3 предложениях>""",

    "Математика": """Ты — опытный учитель математики. Проверь домашнюю работу ученика и дай детальный анализ.

Ответь СТРОГО в следующем формате (каждое поле с новой строки):
GRADE: <число от 1 до 10>
CALCULATION_SCORE: <оценка вычислительных навыков от 1 до 10>
LOGIC_SCORE: <оценка логики решения от 1 до 10>
METHOD_SCORE: <оценка правильности выбора метода от 1 до 10>
PRESENTATION_SCORE: <оценка оформления решения от 1 до 10>
CALCULATION_ERRORS: <список вычислительных ошибок через точку с запятой, или "нет">
LOGIC_ERRORS: <список логических ошибок через точку с запятой, или "нет">
METHOD_ERRORS: <список ошибок в выборе метода через точку с запятой, или "нет">
STRENGTHS: <что сделано хорошо, через точку с запятой>
RECOMMENDATIONS: <конкретные рекомендации по улучшению, через точку с запятой>
LEVEL: <уровень: Начальный/Базовый/Продвинутый/Эксперт>
SUMMARY: <краткое общее заключение в 2-3 предложениях>""",
}


def check_homework(content, api_key, subject="Английский язык"):
    prompt = SUBJECTS.get(subject, SUBJECTS["Английский язык"])
    client = anthropic.Anthropic(api_key=api_key, base_url="https://api.east-api-3.org")

    if content["type"] == "text":
        msg_content = [{"type": "text", "text": prompt + "\n\nРабота ученика:\n" + content["data"]}]
    else:
        msg_content = [{"type": "text", "text": prompt + "\n\nРаспознанный текст с изображения:\n" + content["data"]}]

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": msg_content}],
    )
    result = parse_response(resp.content[0].text)
    result["Предмет"] = subject
    return result