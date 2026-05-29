import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from checker import extract_content, check_homework

GRADES_FILE = "grades.csv"

st.set_page_config(page_title="Проверка работ", page_icon="📝", layout="wide")

# --- Стили ---
st.markdown("""
<style>
.metric-card {background:#f8f9fa;border-radius:10px;padding:16px;text-align:center;border:1px solid #e0e0e0}
.grade-high {color:#28a745;font-size:2.5rem;font-weight:bold}
.grade-mid  {color:#fd7e14;font-size:2.5rem;font-weight:bold}
.grade-low  {color:#dc3545;font-size:2.5rem;font-weight:bold}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input("🔑 Anthropic API Key", type="password", placeholder="sk-ant-...")
    subject = st.selectbox("📚 Предмет", ["Английский язык", "Биология", "Математика"])
    st.divider()
    st.markdown("**Поддерживаемые форматы:**\nPDF, DOCX, JPG, PNG")

if not api_key:
    st.warning("Введите API ключ в боковой панели.")
    st.stop()

# Поля для каждого предмета
SUBJECT_FIELDS = {
    "Английский язык": {
        "score_cols": ["Грамматика", "Лексика", "Орфография", "Связность"],
        "score_keys": ["GRAMMAR_SCORE", "VOCABULARY_SCORE", "SPELLING_SCORE", "COHERENCE_SCORE"],
        "error_sections": [
            ("GRAMMAR_ERRORS", "🔴 Грамматические ошибки"),
            ("SPELLING_ERRORS", "🟠 Орфографические ошибки"),
            ("VOCABULARY_ERRORS", "🟡 Лексические ошибки"),
        ],
    },
    "Биология": {
        "score_cols": ["Знание", "Термины", "Логика", "Примеры"],
        "score_keys": ["KNOWLEDGE_SCORE", "CORRECTNESS_SCORE", "LOGIC_SCORE", "EXAMPLES_SCORE"],
        "error_sections": [
            ("FACTUAL_ERRORS", "🔴 Фактические ошибки"),
            ("TERM_ERRORS", "🟠 Ошибки в терминах"),
            ("STRUCTURE_ERRORS", "🟡 Ошибки в логике"),
        ],
    },
    "Математика": {
        "score_cols": ["Вычисления", "Логика", "Метод", "Оформление"],
        "score_keys": ["CALCULATION_SCORE", "LOGIC_SCORE", "METHOD_SCORE", "PRESENTATION_SCORE"],
        "error_sections": [
            ("CALCULATION_ERRORS", "🔴 Вычислительные ошибки"),
            ("LOGIC_ERRORS", "🟠 Логические ошибки"),
            ("METHOD_ERRORS", "🟡 Ошибки в методе"),
        ],
    },
}

ALL_COLUMNS = ["Файл", "Дата", "Предмет", "Оценка",
               "Грамматика", "Лексика", "Орфография", "Связность",
               "Знание", "Термины", "Логика", "Примеры",
               "Вычисления", "Метод", "Оформление",
               "Уровень", "Ошибки грамматики", "Ошибки орфографии",
               "Ошибки лексики", "Фактические ошибки", "Ошибки терминов",
               "Ошибки логики", "Вычислительные ошибки", "Ошибки метода",
               "Сильные стороны", "Рекомендации", "Заключение"]


# --- Утилиты ---
def load_grades() -> pd.DataFrame:
    if os.path.exists(GRADES_FILE):
        df = pd.read_csv(GRADES_FILE)
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = 0 if "Ошибки" not in col and col not in ["Сильные стороны", "Рекомендации", "Заключение", "Уровень"] else "—"
        num_cols = ["Оценка", "Грамматика", "Лексика", "Орфография", "Связность",
                    "Знание", "Термины", "Логика", "Примеры", "Вычисления", "Метод", "Оформление"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df
    return pd.DataFrame(columns=ALL_COLUMNS)


def save_result(filename: str, r: dict):
    df = load_grades()
    fields = SUBJECT_FIELDS.get(r.get("Предмет", "Английский язык"), SUBJECT_FIELDS["Английский язык"])
    row = {col: 0 for col in ALL_COLUMNS}
    row.update({col: "—" for col in ["Сильные стороны", "Рекомендации", "Заключение", "Уровень"]})
    row["Файл"] = filename
    row["Дата"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    row["Предмет"] = r.get("Предмет", "Английский язык")
    row["Оценка"] = int(r.get("GRADE", 0))
    row["Уровень"] = r.get("LEVEL", "—")
    row["Сильные стороны"] = r.get("STRENGTHS", "—")
    row["Рекомендации"] = r.get("RECOMMENDATIONS", "—")
    row["Заключение"] = r.get("SUMMARY", "—")
    for col_name, key in zip(fields["score_cols"], fields["score_keys"]):
        row[col_name] = int(r.get(key, 0))
    for key, _ in fields["error_sections"]:
        row[key] = r.get(key, "—")
    df.loc[len(df)] = [row.get(c, "—") for c in ALL_COLUMNS]
    df.to_csv(GRADES_FILE, index=False)


def grade_color(g):
    return "grade-high" if g >= 7 else "grade-mid" if g >= 5 else "grade-low"


def radar_chart(r: dict, subj: str):
    fields = SUBJECT_FIELDS.get(subj, SUBJECT_FIELDS["Английский язык"])
    vals = [int(r.get(k, 0)) for k in fields["score_keys"]]
    cats = fields["score_cols"]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                                    fill="toself", line_color="#4e8cff"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10])),
                      margin=dict(l=20, r=20, t=30, b=20), height=300)
    return fig


# === ВКЛАДКИ ===
tab1, tab2 = st.tabs(["📤 Проверка работы", "📊 Аналитика"])

# ── Вкладка 1: Проверка ──
with tab1:
    st.header("Загрузка и проверка работы")
    st.caption(f"Предмет: **{subject}**")
    uploaded = st.file_uploader("Загрузите работу ученика", type=["pdf", "docx", "jpg", "jpeg", "png"])

    if uploaded:
        col_info, col_btn = st.columns([3, 1])
        col_info.success(f"📎 **{uploaded.name}**")

        if col_btn.button("✅ Проверить", type="primary", use_container_width=True):
            with st.spinner("Claude анализирует работу..."):
                try:
                    content = extract_content(uploaded)
                    result = check_homework(content, api_key, subject)
                    save_result(uploaded.name, result)
                    st.session_state["last_result"] = result
                    st.session_state["last_file"] = uploaded.name
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    st.stop()

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        current_subject = r.get("Предмет", "Английский язык")
        fields = SUBJECT_FIELDS.get(current_subject, SUBJECT_FIELDS["Английский язык"])
        st.divider()
        st.subheader(f"Результат: {st.session_state['last_file']} ({current_subject})")

        c1, c2, c3 = st.columns([1, 1, 2])
        grade = int(r.get("GRADE", 0))
        c1.markdown(f"<div class='metric-card'><div style='font-size:.9rem;color:gray'>Итоговая оценка</div>"
                    f"<div class='{grade_color(grade)}'>{grade}/10</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div style='font-size:.9rem;color:gray'>Уровень</div>"
                    f"<div style='font-size:2rem;font-weight:bold'>{r.get('LEVEL','—')}</div></div>",
                    unsafe_allow_html=True)
        c3.plotly_chart(radar_chart(r, current_subject), use_container_width=True)

        st.subheader("📋 Детальный разбор")
        col_l, col_r = st.columns(2)

        with col_l:
            for key, label in fields["error_sections"]:
                with st.expander(label, expanded=True):
                    for e in r.get(key, "нет").split(";"):
                        st.markdown(f"- {e.strip()}")

        with col_r:
            with st.expander("✅ Сильные стороны", expanded=True):
                for s in r.get("STRENGTHS", "—").split(";"):
                    st.markdown(f"- {s.strip()}")
            with st.expander("💡 Рекомендации", expanded=True):
                for rec in r.get("RECOMMENDATIONS", "—").split(";"):
                    st.markdown(f"- {rec.strip()}")

        st.info(f"**Заключение:** {r.get('SUMMARY', '—')}")

# ── Вкладка 2: Аналитика ──
with tab2:
    st.header("Аналитика по всем работам")
    df = load_grades()

    if df.empty:
        st.info("Пока нет проверенных работ.")
    else:
        # Фильтр по предмету
        subjects_in_data = df["Предмет"].unique().tolist()
        filter_subj = st.selectbox("Фильтр по предмету", ["Все предметы"] + subjects_in_data)
        if filter_subj != "Все предметы":
            df = df[df["Предмет"] == filter_subj]

        # Общая статистика
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всего работ", len(df))
        c2.metric("Средний балл", f"{df['Оценка'].mean():.1f}")
        c3.metric("Лучший балл", int(df['Оценка'].max()))
        c4.metric("Худший балл", int(df['Оценка'].min()))

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            fig1 = px.bar(df, x="Файл", y="Оценка", color="Оценка",
                          color_continuous_scale=["#dc3545", "#fd7e14", "#28a745"],
                          range_color=[1, 10], title="Оценки по работам")
            fig1.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig1, use_container_width=True)

        with col_r:
            fig2 = px.pie(df, names="Предмет", values="Оценка", title="Распределение по предметам",
                          hole=0.4)
            fig2.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Детали по выбранной работе
        st.subheader("📄 Детали по работе")
        selected = st.selectbox("Выберите работу", df["Файл"].tolist())
        row = df[df["Файл"] == selected].iloc[-1]
        row_subj = row.get("Предмет", "Английский язык")
        row_fields = SUBJECT_FIELDS.get(row_subj, SUBJECT_FIELDS["Английский язык"])

        score_cols = row_fields["score_cols"]
        metric_cols = st.columns(len(score_cols) + 1)
        metric_cols[0].metric("Оценка", f"{int(row['Оценка'])}/10")
        for i, col in enumerate(score_cols, 1):
            val = int(row.get(col, 0))
            metric_cols[i].metric(col, f"{val}/10")

        col_a, col_b = st.columns(2)
        with col_a:
            for key, label in row_fields["error_sections"]:
                err_val = str(row.get(key, "—"))
                if err_val and err_val != "—" and err_val != "0":
                    st.markdown(f"**{label}:**")
                    for e in err_val.split(";"):
                        if e.strip():
                            st.markdown(f"- {e.strip()}")
        with col_b:
            st.markdown("**✅ Сильные стороны:**")
            for s in str(row.get("Сильные стороны", "—")).split(";"):
                if s.strip():
                    st.markdown(f"- {s.strip()}")
            st.markdown("**💡 Рекомендации:**")
            for rec in str(row.get("Рекомендации", "—")).split(";"):
                if rec.strip():
                    st.markdown(f"- {rec.strip()}")

        st.info(f"**Заключение:** {row.get('Заключение', '—')}")

        st.divider()
        st.subheader("📊 Полная таблица")
        display_cols = ["Файл", "Дата", "Предмет", "Оценка", "Уровень"]
        st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True)