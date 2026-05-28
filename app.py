import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from checker import extract_content, check_homework

GRADES_FILE = "grades.csv"
COLUMNS = ["Файл", "Дата", "Оценка", "Грамматика", "Лексика", "Орфография",
           "Связность", "Уровень", "Ошибки грамматики", "Ошибки орфографии",
           "Ошибки лексики", "Сильные стороны", "Рекомендации", "Заключение"]

st.set_page_config(page_title="Проверка работ по английскому", page_icon="📝", layout="wide")

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
    st.divider()
    st.markdown("**Поддерживаемые форматы:**\nPDF, DOCX, JPG, PNG")

if not api_key:
    st.warning("Введите API ключ в боковой панели.")
    st.stop()


# --- Утилиты ---
def load_grades() -> pd.DataFrame:
    if os.path.exists(GRADES_FILE):
        df = pd.read_csv(GRADES_FILE)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = 0 if col in ["Оценка", "Грамматика", "Лексика", "Орфография", "Связность"] else "—"
        for col in ["Оценка", "Грамматика", "Лексика", "Орфография", "Связность"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df
    return pd.DataFrame(columns=COLUMNS)


def save_result(filename: str, r: dict):
    df = load_grades()
    row = [
        filename,
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        int(r.get("GRADE", 0)),
        int(r.get("GRAMMAR_SCORE", 0)),
        int(r.get("VOCABULARY_SCORE", 0)),
        int(r.get("SPELLING_SCORE", 0)),
        int(r.get("COHERENCE_SCORE", 0)),
        r.get("LEVEL", "—"),
        r.get("GRAMMAR_ERRORS", "—"),
        r.get("SPELLING_ERRORS", "—"),
        r.get("VOCABULARY_ERRORS", "—"),
        r.get("STRENGTHS", "—"),
        r.get("RECOMMENDATIONS", "—"),
        r.get("SUMMARY", "—"),
    ]
    df.loc[len(df)] = row
    df.to_csv(GRADES_FILE, index=False)


def grade_color(g):
    return "grade-high" if g >= 7 else "grade-mid" if g >= 5 else "grade-low"


def radar_chart(r: dict):
    cats = ["Грамматика", "Лексика", "Орфография", "Связность"]
    vals = [int(r.get("GRAMMAR_SCORE", 0)), int(r.get("VOCABULARY_SCORE", 0)),
            int(r.get("SPELLING_SCORE", 0)), int(r.get("COHERENCE_SCORE", 0))]
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
    uploaded = st.file_uploader("Загрузите работу ученика", type=["pdf", "docx", "jpg", "jpeg", "png"])

    if uploaded:
        col_info, col_btn = st.columns([3, 1])
        col_info.success(f"📎 **{uploaded.name}**")

        if col_btn.button("✅ Проверить", type="primary", use_container_width=True):
            with st.spinner("Claude анализирует работу..."):
                try:
                    content = extract_content(uploaded)
                    result = check_homework(content, api_key)
                    save_result(uploaded.name, result)
                    st.session_state["last_result"] = result
                    st.session_state["last_file"] = uploaded.name
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    st.stop()

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.divider()
        st.subheader(f"Результат: {st.session_state['last_file']}")

        # Оценка + уровень
        c1, c2, c3 = st.columns([1, 1, 2])
        grade = int(r.get("GRADE", 0))
        c1.markdown(f"<div class='metric-card'><div style='font-size:.9rem;color:gray'>Итоговая оценка</div>"
                    f"<div class='{grade_color(grade)}'>{grade}/10</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div style='font-size:.9rem;color:gray'>Уровень</div>"
                    f"<div style='font-size:2rem;font-weight:bold'>{r.get('LEVEL','—')}</div></div>",
                    unsafe_allow_html=True)
        c3.plotly_chart(radar_chart(r), use_container_width=True)

        # Детали
        st.subheader("📋 Детальный разбор")
        col_l, col_r = st.columns(2)

        with col_l:
            with st.expander("🔴 Грамматические ошибки", expanded=True):
                for e in r.get("GRAMMAR_ERRORS", "нет").split(";"):
                    st.markdown(f"- {e.strip()}")
            with st.expander("🟠 Орфографические ошибки", expanded=True):
                for e in r.get("SPELLING_ERRORS", "нет").split(";"):
                    st.markdown(f"- {e.strip()}")
            with st.expander("🟡 Лексические ошибки", expanded=True):
                for e in r.get("VOCABULARY_ERRORS", "нет").split(";"):
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
        # Общая статистика
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Всего работ", len(df))
        c2.metric("Средний балл", f"{df['Оценка'].mean():.1f}")
        c3.metric("Лучший балл", int(df['Оценка'].max()))
        c4.metric("Худший балл", int(df['Оценка'].min()))
        c5.metric("Самый частый уровень", df['Уровень'].mode()[0] if 'Уровень' in df else "—")

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            # График оценок по работам
            fig1 = px.bar(df, x="Файл", y="Оценка", color="Оценка",
                          color_continuous_scale=["#dc3545", "#fd7e14", "#28a745"],
                          range_color=[1, 10], title="Оценки по работам")
            fig1.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig1, use_container_width=True)

        with col_r:
            # Средние баллы по критериям
            score_cols = ["Грамматика", "Лексика", "Орфография", "Связность"]
            avgs = [df[c].mean() for c in score_cols if c in df.columns]
            fig2 = go.Figure(go.Bar(x=score_cols, y=avgs, marker_color="#4e8cff"))
            fig2.update_layout(title="Средние баллы по критериям",
                               yaxis_range=[0, 10], height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Детальная таблица по каждой работе
        st.subheader("📄 Детали по каждой работе")
        selected = st.selectbox("Выберите работу", df["Файл"].tolist())
        row = df[df["Файл"] == selected].iloc[-1]

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Оценка", f"{int(row['Оценка'])}/10")
        d2.metric("Грамматика", f"{int(row['Грамматика'])}/10")
        d3.metric("Лексика", f"{int(row['Лексика'])}/10")
        d4.metric("Орфография", f"{int(row['Орфография'])}/10")
        d5.metric("Связность", f"{int(row['Связность'])}/10")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔴 Ошибки грамматики:**")
            for e in str(row.get("Ошибки грамматики", "—")).split(";"):
                st.markdown(f"- {e.strip()}")
            st.markdown("**🟠 Ошибки орфографии:**")
            for e in str(row.get("Ошибки орфографии", "—")).split(";"):
                st.markdown(f"- {e.strip()}")
        with col_b:
            st.markdown("**✅ Сильные стороны:**")
            for s in str(row.get("Сильные стороны", "—")).split(";"):
                st.markdown(f"- {s.strip()}")
            st.markdown("**💡 Рекомендации:**")
            for rec in str(row.get("Рекомендации", "—")).split(";"):
                st.markdown(f"- {rec.strip()}")

        st.info(f"**Заключение:** {row.get('Заключение', '—')}")

        st.divider()
        st.subheader("📊 Полная таблица")
        st.dataframe(df[["Файл", "Дата", "Оценка", "Грамматика", "Лексика",
                          "Орфография", "Связность", "Уровень"]], use_container_width=True)
