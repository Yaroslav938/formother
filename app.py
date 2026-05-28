import os
import streamlit as st
import pandas as pd
from checker import extract_content, check_homework

GRADES_FILE = "grades.csv"

st.set_page_config(page_title="Проверка работ по английскому", page_icon="📝")
st.title("📝 Проверка домашних работ по английскому")

with st.sidebar:
    api_key = st.text_input("🔑 Anthropic API Key", type="password", placeholder="sk-ant-...")

if not api_key:
    st.warning("Введите API ключ в боковой панели.")
    st.stop()


def load_grades() -> pd.DataFrame:
    if os.path.exists(GRADES_FILE):
        return pd.read_csv(GRADES_FILE)
    return pd.DataFrame(columns=["Файл", "Оценка", "Дата"])


def save_grade(filename: str, grade: int):
    df = load_grades()
    df.loc[len(df)] = [filename, grade, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")]
    df.to_csv(GRADES_FILE, index=False)


# --- Загрузка файла ---
uploaded = st.file_uploader(
    "Загрузите работу ученика",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
)

if uploaded:
    st.success(f"Файл загружен: **{uploaded.name}**")

    if st.button("✅ Проверить работу", type="primary"):
        with st.spinner("Claude анализирует работу..."):
            try:
                content = extract_content(uploaded)
                feedback, grade = check_homework(content, api_key)
                save_grade(uploaded.name, grade)
            except Exception as e:
                st.error(f"Ошибка: {e}")
                st.stop()

        color = "green" if grade >= 7 else "orange" if grade >= 5 else "red"
        st.markdown(f"### Оценка: :{color}[{grade}/10]")

        sections = {"ERRORS": "🔴 Ошибки", "FEEDBACK": "💡 Рекомендации"}
        lines = {l.split(":")[0]: ":".join(l.split(":")[1:]).strip()
                 for l in feedback.splitlines() if ":" in l}

        for key, label in sections.items():
            if key in lines:
                st.markdown(f"**{label}:** {lines[key]}")

# --- Аналитика ---
st.divider()
st.subheader("📊 Аналитика")

df = load_grades()
if df.empty:
    st.info("Пока нет проверенных работ.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего работ", len(df))
    col2.metric("Средний балл", f"{df['Оценка'].mean():.1f}")
    col3.metric("Лучший балл", int(df['Оценка'].max()))

    st.bar_chart(df.set_index("Файл")["Оценка"])
    st.dataframe(df, use_container_width=True)
