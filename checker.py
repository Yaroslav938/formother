import base64
import anthropic
from pypdf import PdfReader
from docx import Document

PROMPT = """Ты — опытный учитель английского языка. Проверь домашнюю работу ученика и дай детальный анализ.

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
SUMMARY: <краткое общее заключение в 2-3 предложениях>"""


def extract_content(file) -> dict:
    ext = file.name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        text = "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
        return {"type": "text", "data": text}
    elif ext == "docx":
        text = "\n".join(p.text for p in Document(file).paragraphs)
        return {"type": "text", "data": text}
    else:
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        b64 = base64.b64encode(file.read()).decode()
        return {"type": "image", "data": b64, "media_type": mime}


def parse_response(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def check_homework(content: dict, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key, base_url="https://api.east-api-3.org")

    if content["type"] == "text":
        msg_content = [{"type": "text", "text": PROMPT + "\n\nРабота ученика:\n" + content["data"]}]
    else:
        msg_content = [
            {"type": "image", "source": {"type": "base64",
             "media_type": content["media_type"], "data": content["data"]}},
            {"type": "text", "text": PROMPT},
        ]

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": msg_content}],
    )
    return parse_response(resp.content[0].text)
