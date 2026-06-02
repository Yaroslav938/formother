"""
app.py — Streamlit-приложение проверки работ учеников через Claude.

Доработки:
- Динамическая схема хранения: столбцы CSV формируются из конфигурации предметов
  (никаких жёстко зашитых списков → устойчиво к добавлению предметов).
- 6 предметов: Английский, Русский, Биология, Математика, Физика, Химия.
- Расширенная аналитика: KPI, тренд по датам, средние по критериям, частые ошибки,
  сравнение предметов, фильтры по дате/предмету, экспорт в CSV.
- Современный UI: кастомный CSS, карточки, аккуратные секции, понятная навигация.
"""

import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from checker import (
    extract_content,
    check_homework,
    diagnose_vision,
    ocr_available,
    get_subjects,
    get_subject_config,
)

GRADES_FILE = "grades.csv"

st.set_page_config(page_title="Проверка работ • AI", page_icon="📝", layout="wide")

# ============================================================
#  Стили (современный, чистый интерфейс)
# ============================================================
st.markdown("""
<style>
:root {
  --accent:#4e8cff;
  --bg-card:#ffffff;
  --border:#e6e9ef;
}
.main .block-container {padding-top:1.5rem; max-width:1300px;}
h1,h2,h3 {letter-spacing:-0.01em;}

/* Карточки метрик */
.metric-card{
  background:var(--bg-card); border-radius:14px; padding:18px 16px;
  text-align:center; border:1px solid var(--border);
  box-shadow:0 1px 3px rgba(16,24,40,.04);
}
.metric-card .label{font-size:.8rem; color:#667085; text-transform:uppercase; letter-spacing:.04em;}
.grade-high{color:#12b76a; font-size:2.6rem; font-weight:800; line-height:1.1;}
.grade-mid {color:#f79009; font-size:2.6rem; font-weight:800; line-height:1.1;}
.grade-low {color:#f04438; font-size:2.6rem; font-weight:800; line-height:1.1;}
.level-badge{font-size:1.9rem; font-weight:800; color:#1d2939;}

/* Заголовок-герой */
.hero{
  background:linear-gradient(135deg,#4e8cff 0%,#7b61ff 100%);
  color:#fff; border-radius:16px; padding:22px 26px; margin-bottom:14px;
}
.hero h1{color:#fff; margin:0; font-size:1.7rem;}
.hero p{color:#eaf0ff; margin:.3rem 0 0; font-size:.95rem;}

/* Чип предмета */
.subj-chip{
  display:inline-block; background:#eef4ff; color:#2563eb;
  padding:4px 12px; border-radius:999px; font-weight:600; font-size:.85rem;
}

/* Секции ошибок */
.section-title{font-weight:700; margin:.4rem 0 .2rem; font-size:1rem;}
div[data-testid="stExpander"]{border-radius:12px; border:1px solid var(--border);}
.stMetric{background:#fafbfc;border:1px solid var(--border);border-radius:12px;padding:8px 10px;}
</style>
""", unsafe_allow_html=True)


# ============================================================
#  Динамическая схема столбцов
# ============================================================
SUBJECTS = get_subjects()

# Базовые столбцы (общие для всех предметов)
BASE_COLUMNS = ["Файл", "Дата", "Предмет", "Оценка", "Уровень",
                "Сильные стороны", "Рекомендации", "Заключение"]

# Все score-ключи и error-ключи по всем предметам (для единой ширины CSV)
ALL_SCORE_KEYS = []
ALL_ERROR_KEYS = []
SCORE_LABELS = {}   # ключ -> человекочитаемое имя
ERROR_LABELS = {}   # ключ -> заголовок секции
for subj in SUBJECTS:
    cfg = get_subject_config(subj)
    for k, label in cfg["scores"]:
        if k not in ALL_SCORE_KEYS:
            ALL_SCORE_KEYS.append(k)
            SCORE_LABELS[k] = label
    for k, label in cfg["errors"]:
        if k not in ALL_ERROR_KEYS:
            ALL_ERROR_KEYS.append(k)
            ERROR_LABELS[k] = label

ALL_COLUMNS = BASE_COLUMNS + ALL_SCORE_KEYS + ALL_ERROR_KEYS
NUM_COLUMNS = ["Оценка"] + ALL_SCORE_KEYS
TEXT_DEFAULT_DASH = ["Уровень", "Сильные стороны", "Рекомендации", "Заключение"] + ALL_ERROR_KEYS


# ============================================================
#  Утилиты хранения
# ============================================================
def load_grades() -> pd.DataFrame:
    if os.path.exists(GRADES_FILE):
        df = pd.read_csv(GRADES_FILE)
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = "—" if col in TEXT_DEFAULT_DASH else 0
        for col in NUM_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        # Парсим дату для аналитики
        df["_dt"] = pd.to_datetime(df["Дата"], errors="coerce")
        return df
    return pd.DataFrame(columns=ALL_COLUMNS)


def save_result(filename: str, r: dict):
    df = load_grades()
    if "_dt" in df.columns:
        df = df.drop(columns=["_dt"])
    subj = r.get("Предмет", "Английский язык")
    cfg = get_subject_config(subj)

    row = {c: 0 for c in ALL_COLUMNS}
    for c in TEXT_DEFAULT_DASH:
        row[c] = "—"

    row["Файл"] = filename
    row["Дата"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    row["Предмет"] = subj
    row["Оценка"] = int(r.get("GRADE", 0))
    row["Уровень"] = r.get("LEVEL", "—")
    row["Сильные стороны"] = r.get("STRENGTHS", "—")
    row["Рекомендации"] = r.get("RECOMMENDATIONS", "—")
    row["Заключение"] = r.get("SUMMARY", "—")
    for k, _ in cfg["scores"]:
        row[k] = int(r.get(k, 0))
    for k, _ in cfg["errors"]:
        row[k] = r.get(k, "нет")

    df.loc[len(df)] = [row.get(c, "—") for c in ALL_COLUMNS]
    df.to_csv(GRADES_FILE, index=False)


def grade_color(g: int) -> str:
    return "grade-high" if g >= 7 else "grade-mid" if g >= 5 else "grade-low"


def radar_chart(r: dict, subj: str):
    cfg = get_subject_config(subj)
    keys = [k for k, _ in cfg["scores"]]
    cats = [lbl for _, lbl in cfg["scores"]]
    vals = [int(r.get(k, 0)) for k in keys]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", line_color="#4e8cff", fillcolor="rgba(78,140,255,.25)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10], tickfont=dict(size=10))),
        margin=dict(l=30, r=30, t=30, b=20), height=300, showlegend=False)
    return fig


def split_items(value: str):
    """Разбивает строку ошибок/рекомендаций на список непустых пунктов."""
    s = str(value or "").strip()
    if not s or s in ("—", "0", "нет", "Нет", "None"):
        return []
    items = [x.strip(" .;") for x in s.replace("\n", ";").split(";")]
    return [x for x in items if x and x.lower() != "нет"]


# ============================================================
#  Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input("🔑 Anthropic API Key", type="password", placeholder="sk-ant-...")
    subject = st.selectbox("📚 Предмет", SUBJECTS)
    st.divider()
    st.markdown("**Поддерживаемые форматы:**\n\nPDF, DOCX, JPG, PNG")
    st.caption("Изображения отправляются в Claude Vision; для фото/сканов также "
               "выполняется локальное распознавание текста (если доступно).")
    st.divider()
    if ocr_available():
        st.success("✅ OCR доступен")
    else:
        st.warning("⚠\ufe0f OCR недоступен (фото не распознаётся)")
    debug_mode = st.checkbox("🐞 Режим диагностики",
                            help="Показывает технические детали: что уходит в модель и что она вернула.")
    if st.button("🔍 Проверить поддержку изображений", use_container_width=True):
        with st.spinner("Проверяю, видит ли модель картинки..."):
            diag = diagnose_vision(api_key)
        if diag.get("error"):
            st.error(f"Ошибка запроса: {diag['error']}")
        elif diag.get("vision_ok"):
            st.success("✅ Изображения работают: модель прочитала текст с картинки.")
        else:
            st.error("❌ Модель НЕ видит картинки. Вероятно, прокси не поддерживает vision "
                     "или подменяет модель. С фото будет работать только OCR-режим.")
        st.json(diag)
    st.divider()
    df_side = load_grades()
    if not df_side.empty:
        st.metric("Проверено работ", len(df_side))

if not api_key:
    st.markdown("""
    <div class="hero">
      <h1>📝 AI-проверка работ учеников</h1>
      <p>Загрузите фото или документ — Claude проверит работу, выставит оценку и даст разбор по критериям.</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning("Введите Anthropic API ключ в боковой панели, чтобы начать.")
    st.stop()


# ============================================================
#  Вкладки
# ============================================================
tab1, tab2 = st.tabs(["📤 Проверка работы", "📊 Аналитика"])

# ── Вкладка 1: Проверка ──────────────────────────────────
with tab1:
    st.markdown(f"""
    <div class="hero">
      <h1>Проверка работы</h1>
      <p>Текущий предмет: <span style="font-weight:700">{subject}</span></p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Загрузите работу ученика",
        type=["pdf", "docx", "jpg", "jpeg", "png"],
        help="Принимаются PDF, DOCX и изображения (фото тетради тоже подойдут).")

    if uploaded:
        col_info, col_btn = st.columns([3, 1])
        col_info.success(f"📎 **{uploaded.name}**")
        run = col_btn.button("✅ Проверить", type="primary", use_container_width=True)

        if run:
            with st.spinner("Claude анализирует работу..."):
                try:
                    content = extract_content(uploaded)
                    if content["type"] == "text" and len(content.get("data", "").strip()) < 5:
                        st.error("Не удалось извлечь текст из файла. "
                                 "Попробуйте загрузить более чёткое фото или другой формат.")
                        st.stop()

                    # Для изображений: прокси не передаёт картинки, поэтому нужен OCR-текст.
                    # Если OCR недоступен или ничего не распознал — честно предупреждаем.
                    if content["type"] == "image" and len((content.get("ocr") or "").strip()) < 5:
                        if not ocr_available():
                            st.error(
                                "На сервере недоступен OCR, а текущий API-прокси не передаёт "
                                "изображения в модель. Добавьте в requirements.txt строку "
                                "`rapidocr-onnxruntime` (работает без системных пакетов) и "
                                "перезапустите приложение, либо загружайте работы в формате PDF/DOCX с текстом.")
                        else:
                            st.error(
                                "Не удалось распознать текст на фото. Сделайте снимок чётче, "
                                "при хорошем освещении и без наклона. Текущий API-прокси не видит "
                                "изображения напрямую, поэтому нужен распознаваемый текст.")
                        st.stop()

                    if content["type"] == "image" and (content.get("ocr") or "").strip():
                        st.caption("ℹ\ufe0f Текущий прокси не принимает изображения — проверка выполняется "
                                   "по тексту, распознанному локально (OCR).")

                    result = check_homework(content, api_key, subject, debug=debug_mode)

                    if debug_mode and result.get("_debug"):
                        with st.expander("🐞 Диагностика запроса", expanded=True):
                            st.json(result["_debug"])

                    if result.get("_parse_failed"):
                        st.error("Не удалось разобрать ответ модели. Показываю сырой ответ ниже.")
                        with st.expander("Сырой ответ модели"):
                            st.code(result.get("_raw", ""))
                        st.stop()

                    save_result(uploaded.name, result)
                    st.session_state["last_result"] = result
                    st.session_state["last_file"] = uploaded.name
                except Exception as e:
                    st.error(f"Ошибка при проверке: {e}")
                    st.stop()

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        current_subject = r.get("Предмет", subject)
        cfg = get_subject_config(current_subject)

        st.divider()
        st.markdown(
            f"### Результат: {st.session_state['last_file']} "
            f"<span class='subj-chip'>{current_subject}</span>",
            unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 2])
        grade = int(r.get("GRADE", 0))
        c1.markdown(
            f"<div class='metric-card'><div class='label'>Итоговая оценка</div>"
            f"<div class='{grade_color(grade)}'>{grade}/10</div></div>",
            unsafe_allow_html=True)
        c2.markdown(
            f"<div class='metric-card'><div class='label'>Уровень</div>"
            f"<div class='level-badge'>{r.get('LEVEL', '—')}</div></div>",
            unsafe_allow_html=True)
        c3.plotly_chart(radar_chart(r, current_subject), use_container_width=True)

        # Полоски по критериям
        st.markdown("#### Оценки по критериям")
        score_cols = st.columns(len(cfg["scores"]))
        for i, (k, label) in enumerate(cfg["scores"]):
            v = int(r.get(k, 0))
            score_cols[i].metric(label, f"{v}/10")
            score_cols[i].progress(v / 10)

        st.subheader("📋 Детальный разбор")
        col_l, col_r = st.columns(2)
        with col_l:
            for key, label in cfg["errors"]:
                items = split_items(r.get(key, "нет"))
                with st.expander(f"{label} ({len(items)})", expanded=bool(items)):
                    if items:
                        for e in items:
                            st.markdown(f"- {e}")
                    else:
                        st.markdown("✅ Ошибок не найдено")
        with col_r:
            with st.expander("✅ Сильные стороны", expanded=True):
                items = split_items(r.get("STRENGTHS", "—"))
                for s in (items or ["—"]):
                    st.markdown(f"- {s}")
            with st.expander("💡 Рекомендации", expanded=True):
                items = split_items(r.get("RECOMMENDATIONS", "—"))
                for rec in (items or ["—"]):
                    st.markdown(f"- {rec}")

        st.info(f"**Заключение:** {r.get('SUMMARY', '—')}")


# ── Вкладка 2: Аналитика ─────────────────────────────────
with tab2:
    st.markdown("""
    <div class="hero">
      <h1>Аналитика</h1>
      <p>Сводка по всем проверенным работам: тренды, средние по критериям и частые ошибки.</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_grades()
    if df.empty:
        st.info("Пока нет проверенных работ. Проверьте первую работу во вкладке «Проверка работы».")
    else:
        # --- Фильтры ---
        fc1, fc2 = st.columns([2, 3])
        subjects_in_data = sorted(df["Предмет"].dropna().unique().tolist())
        filter_subj = fc1.selectbox("Предмет", ["Все предметы"] + subjects_in_data)

        valid_dates = df["_dt"].dropna()
        if not valid_dates.empty:
            min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
            date_range = fc2.date_input("Период", value=(min_d, max_d),
                                        min_value=min_d, max_value=max_d)
        else:
            date_range = None

        fdf = df.copy()
        if filter_subj != "Все предметы":
            fdf = fdf[fdf["Предмет"] == filter_subj]
        if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
            fdf = fdf[(fdf["_dt"].dt.date >= start) & (fdf["_dt"].dt.date <= end)]

        if fdf.empty:
            st.warning("Нет данных под выбранные фильтры.")
            st.stop()

        # --- KPI ---
        st.markdown("#### Ключевые показатели")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Всего работ", len(fdf))
        k2.metric("Средний балл", f"{fdf['Оценка'].mean():.1f}")
        k3.metric("Медиана", f"{fdf['Оценка'].median():.0f}")
        k4.metric("Лучший", int(fdf["Оценка"].max()))
        k5.metric("Худший", int(fdf["Оценка"].min()))

        st.divider()

        # --- Ряд 1: распределение оценок + по предметам ---
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig_hist = px.histogram(fdf, x="Оценка", nbins=10, title="Распределение оценок",
                                    color_discrete_sequence=["#4e8cff"])
            fig_hist.update_layout(height=320, margin=dict(t=45, b=10), bargap=0.1,
                                   xaxis=dict(dtick=1, range=[0, 10.5]))
            st.plotly_chart(fig_hist, use_container_width=True)
        with r1c2:
            by_subj = fdf.groupby("Предмет").agg(
                Средний=("Оценка", "mean"), Работ=("Оценка", "count")).reset_index()
            fig_subj = px.bar(by_subj, x="Предмет", y="Средний", color="Средний",
                              color_continuous_scale=["#f04438", "#f79009", "#12b76a"],
                              range_color=[1, 10], title="Средний балл по предметам",
                              text=by_subj["Средний"].round(1))
            fig_subj.update_layout(height=320, margin=dict(t=45, b=10),
                                   yaxis=dict(range=[0, 10]))
            st.plotly_chart(fig_subj, use_container_width=True)

        # --- Ряд 2: динамика по времени ---
        st.markdown("#### Динамика оценок во времени")
        tdf = fdf.dropna(subset=["_dt"]).sort_values("_dt")
        if len(tdf) >= 2:
            fig_time = px.line(tdf, x="_dt", y="Оценка", markers=True,
                               color="Предмет" if filter_subj == "Все предметы" else None)
            fig_time.update_layout(height=320, margin=dict(t=20, b=10),
                                   yaxis=dict(range=[0, 10.5]), xaxis_title="Дата")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.caption("Для графика динамики нужно минимум 2 работы.")

        # --- Ряд 3: средние по критериям (для конкретного предмета) ---
        if filter_subj != "Все предметы":
            cfg = get_subject_config(filter_subj)
            st.markdown(f"#### Средние по критериям — {filter_subj}")
            keys = [k for k, _ in cfg["scores"]]
            labels = [lbl for _, lbl in cfg["scores"]]
            means = [fdf[k].mean() if k in fdf.columns else 0 for k in keys]
            crit_df = pd.DataFrame({"Критерий": labels, "Средний балл": means})
            fig_crit = px.bar(crit_df, x="Средний балл", y="Критерий", orientation="h",
                              color="Средний балл", range_color=[1, 10],
                              color_continuous_scale=["#f04438", "#f79009", "#12b76a"],
                              text=crit_df["Средний балл"].round(1))
            fig_crit.update_layout(height=300, margin=dict(t=10, b=10),
                                   xaxis=dict(range=[0, 10]))
            st.plotly_chart(fig_crit, use_container_width=True)

            # --- Частые ошибки ---
            st.markdown("#### Самые частые ошибки")
            error_counter = {}
            for k, label in cfg["errors"]:
                if k not in fdf.columns:
                    continue
                for _, val in fdf[k].items():
                    for item in split_items(val):
                        error_counter[item] = error_counter.get(item, 0) + 1
            if error_counter:
                top = sorted(error_counter.items(), key=lambda x: -x[1])[:10]
                err_df = pd.DataFrame(top, columns=["Ошибка", "Частота"])
                fig_err = px.bar(err_df, x="Частота", y="Ошибка", orientation="h",
                                 color_discrete_sequence=["#f04438"])
                fig_err.update_layout(height=max(250, 35 * len(top)),
                                      margin=dict(t=10, b=10),
                                      yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_err, use_container_width=True)
            else:
                st.caption("Ошибок в выбранных работах не зафиксировано.")
        else:
            st.caption("Выберите конкретный предмет в фильтре, чтобы увидеть разбор по критериям и частые ошибки.")

        st.divider()

        # --- Детали по работе ---
        st.subheader("📄 Детали по работе")
        sel = st.selectbox("Выберите работу", fdf["Файл"].tolist())
        row = fdf[fdf["Файл"] == sel].iloc[-1]
        row_subj = row.get("Предмет", "Английский язык")
        row_cfg = get_subject_config(row_subj)

        mcols = st.columns(len(row_cfg["scores"]) + 1)
        mcols[0].metric("Оценка", f"{int(row['Оценка'])}/10")
        for i, (k, label) in enumerate(row_cfg["scores"], 1):
            mcols[i].metric(label, f"{int(row.get(k, 0))}/10")

        da, db = st.columns(2)
        with da:
            for key, label in row_cfg["errors"]:
                items = split_items(row.get(key, "нет"))
                if items:
                    st.markdown(f"<div class='section-title'>{label}</div>", unsafe_allow_html=True)
                    for e in items:
                        st.markdown(f"- {e}")
        with db:
            st.markdown("<div class='section-title'>✅ Сильные стороны</div>", unsafe_allow_html=True)
            for s in (split_items(row.get("Сильные стороны", "—")) or ["—"]):
                st.markdown(f"- {s}")
            st.markdown("<div class='section-title'>💡 Рекомендации</div>", unsafe_allow_html=True)
            for rec in (split_items(row.get("Рекомендации", "—")) or ["—"]):
                st.markdown(f"- {rec}")

        st.info(f"**Заключение:** {row.get('Заключение', '—')}")

        st.divider()

        # --- Таблица + экспорт ---
        st.subheader("📊 Полная таблица")
        display_cols = ["Файл", "Дата", "Предмет", "Оценка", "Уровень"]
        st.dataframe(fdf[[c for c in display_cols if c in fdf.columns]],
                     use_container_width=True, hide_index=True)

        export_df = fdf.drop(columns=[c for c in ["_dt"] if c in fdf.columns])
        st.download_button(
            "⬇️ Скачать данные (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="grades_export.csv",
            mime="text/csv")
