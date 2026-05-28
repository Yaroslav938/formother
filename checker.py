import base64
import anthropic
from pypdf import PdfReader
from docx import Document

PROMPT = """Ты — учитель английского языка. Проверь домашнюю работу ученика.

Ответь строго в таком формате:
GRADE: <число от 1 до 10>
ERRORS: <список ошибок через точку с запятой, или "нет ошибок">
FEEDBACK: <краткие рекомендации по улучшению>"""


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


def check_homework(content: dict, api_key: str) -> tuple[str, int]:
    client = anthropic.Anthropic(api_key=api_key)

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
    text = resp.content[0].text

    grade_line = next((l for l in text.splitlines() if l.startswith("GRADE:")), "GRADE: 0")
    grade = int(grade_line.split(":")[1].strip())

    return text, grade
