# analitika_kk.py
"""
Streamlit қосымшасы — JSON-файлдан толық аналитика (барлығы қазақша).
Қателер түзетілді: parsed_date, date_input фильтрлері, бос мәтіндер.
"""

import json
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import CountVectorizer
DetectorFactory.seed = 0

import pycountry

# ---------------------------
# Қазақша мәтіндер
# ---------------------------
KK = {
    "title": "Деректер аналитикасы",
    "subtitle": "Толық деректер файлын жүктеп, графиктер мен статистикаларды көру",
    "upload_prompt": "JSON файлын таңдаңыз (немесе төмендегі жолды пайдаланыңыз):",
    "local_path_prompt": "Жергілікті файл жолын көрсету (опционалды):",
    "use_local_button": "Жергілікті файлды жүктеу",
    "no_file": "Файл табылған жоқ. Жүктеңіз немесе жол енгізіңіз.",
    "overview": "Жалпы шолу",
    "total_records": "Жалпы жазба саны",
    "filtered_records": "Сүзгіленген жазбалар",
    "date_range": "Файлдағы күндер аралығы",
    "avg_len": "Орташа мәтін ұзындығы (символдар)",
    "time_series": "Уақыт бойынша тренд",
    "time_series_expl": "Күн бойынша жазбалар санының динамикасы",
    "top_sources": "Топ дереккөздер (source)",
    "top_sources_expl": "Ең көп жазба түскен дереккөздер",
    "text_length": "Мәтін ұзындығының таралуы",
    "text_length_expl": "Мәтін ұзындығының таралу гистограммасы",
    "annotations": "Аннотациялар — меткалар (labels) және таңдаулар (choices)",
    "annotations_expl": "Аннотациялар жиілігі мен choice-тар таралуын көрсетеді",
    "wordcloud": "WordCloud — жиі кездесетін сөздер",
    "wordcloud_expl": "Мәтіндерден алынған жиі кездесетін сөздер",
    "top_tokens": "Топ сөздер",
    "save_csv": "Сүзілген нәтижені CSV-ге сақтау",
    "csv_saved": "CSV сақталды:",
    "download_csv": "CSV жүктеу сілтемесі дайын",
    "no_text": "Мәтін өрістері бос — WordCloud және токендер шығарылмайды",
    "select_filters": "Фильтрлер және параметрлер",
    "sources_filter": "Дереккөздерді таңдау",
    "date_filter": "Күн аралығын таңдаңыз",
    "load_error": "JSON оқу қателігі: ",
    "ready": "Қош келдіңіз, аналитика дайын!"
}

# ---------------------------
# Файл жүктеу / оқу
# ---------------------------
def safe_load_json_file(file_like):
    try:
        return json.load(file_like)
    except Exception as e:
        raise

def normalize_records(raw):
    records = []
    if isinstance(raw, dict) and not isinstance(raw, list):
        if "data" in raw and isinstance(raw["data"], dict):
            records = [raw]
        else:
            records = list(raw.values())
    elif isinstance(raw, list):
        records = raw
    else:
        records = [raw]

    out_rows = []
    for rec in records:
        data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        source = data.get("source") or rec.get("source") or (rec.get("meta") or {}).get("source")
        date = data.get("date") or rec.get("date") or (rec.get("meta") or {}).get("date")
        text = data.get("text") or rec.get("text") or rec.get("content") or (data.get("body") if isinstance(data.get("body"), str) else "")
        annotations = rec.get("annotations") or data.get("annotations") or rec.get("labels") or []
        if isinstance(text, dict):
            for key in ("kk","kz","ru","en"):
                if key in text:
                    text = text[key]
                    break
        out_rows.append({"source": source, "date": date, "text": text, "annotations": annotations})
    return out_rows

def parse_date_safe(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return pd.NaT
    try:
        return pd.to_datetime(d).tz_localize(None)
    except Exception:
        try:
            s = str(d).replace("T"," ").replace("Z","")
            return pd.to_datetime(s).tz_localize(None)
        except Exception:
            return pd.NaT

def extract_annotation_info(anns):
    labels = []
    choices = Counter()
    if not anns:
        return labels, choices
    if isinstance(anns, dict):
        anns = [anns]
    for a in anns:
        if not isinstance(a, dict):
            continue
        results = a.get("result") or a.get("value") or []
        if isinstance(results, dict):
            results = [results]
        for r in results:
            if isinstance(r.get("value"), dict):
                v = r.get("value")
                if "labels" in v:
                    labels.extend(v["labels"])
                if "choices" in v:
                    for ch in v["choices"]:
                        choices[ch] += 1
            if a.get("label"):
                labels.append(a.get("label"))
    return labels, choices

def detect_language_safe(text):
    try:
        if not text or len(text.strip())<5:
            return "unknown"
        return detect(text)
    except:
        return "unknown"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=KK["title"], layout="wide")
st.title(KK["title"])
st.caption(KK["subtitle"])

JSON_PATH = Path("project.json")
raw_data = None
if JSON_PATH.exists():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        st.success(f"{KK['ready']} (автоматты түрде {JSON_PATH} жүктелді)")
    except Exception as e:
        st.warning(KK["load_error"] + str(e))

if raw_data is None:
    uploaded = st.file_uploader(KK["upload_prompt"], type=["json"])
    local_path = st.text_input(KK["local_path_prompt"])
    if uploaded is not None:
        raw_data = safe_load_json_file(uploaded)
    elif local_path:
        p = Path(local_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            st.warning(KK["no_file"])
            st.stop()
    else:
        st.info(KK["no_file"])
        st.stop()

# DataFrame
rows = normalize_records(raw_data)
df = pd.DataFrame(rows)
df["parsed_date"] = df["date"].apply(parse_date_safe)
df["text_len"] = df["text"].astype(str).apply(len)

# Sidebar фильтры
st.sidebar.header(KK["select_filters"])
sources_opts = sorted(df["source"].dropna().unique()) if not df["source"].isna().all() else []
src_filter = st.sidebar.multiselect(KK["sources_filter"], options=sources_opts, default=None)
min_date = df["parsed_date"].min()
max_date = df["parsed_date"].max()
default_range = (min_date.date() if pd.notnull(min_date) else None,
                 max_date.date() if pd.notnull(max_date) else None)
date_range = st.sidebar.date_input(KK["date_filter"], value=default_range)

# Checkbox
show_overview = st.sidebar.checkbox(KK["overview"], True)
show_time = st.sidebar.checkbox(KK["time_series"], True)
show_sources = st.sidebar.checkbox(KK["top_sources"], True)
show_lengths = st.sidebar.checkbox(KK["text_length"], True)
show_annotations = st.sidebar.checkbox(KK["annotations"], True)
show_wordcloud = st.sidebar.checkbox(KK["wordcloud"], True)

# Filtered DF
filtered = df.copy()
if src_filter:
    filtered = filtered[filtered["source"].isin(src_filter)]
if isinstance(date_range, tuple) and len(date_range)==2:
    start_d, end_d = date_range
    if start_d:
        filtered = filtered[filtered["parsed_date"] >= pd.to_datetime(start_d)]
    if end_d:
        filtered = filtered[filtered["parsed_date"] <= pd.to_datetime(end_d) + pd.Timedelta(days=1)]

# Overview
if show_overview:
    st.subheader(KK["overview"])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(KK["total_records"], len(df))
    c2.metric(KK["filtered_records"], len(filtered))
    c3.metric(KK["date_range"], f"{min_date.date() if pd.notnull(min_date) else '—'} — {max_date.date() if pd.notnull(max_date) else '—'}")
    c4.metric(KK["avg_len"], int(df["text_len"].mean()) if len(df)>0 else 0)

# Time series
if show_time:
    st.subheader(KK["time_series"])
    st.caption(KK["time_series_expl"])
    ts = filtered.copy()
    ts["day"] = ts["parsed_date"].dt.floor("d")
    ts_counts = ts.groupby("day").size().reset_index(name="count").dropna()
    if ts_counts.empty:
        st.info("Күндік деректер жоқ")
    else:
        fig = px.line(ts_counts, x="day", y="count", title=KK["time_series"])
        st.plotly_chart(fig, use_container_width=True)

# Top sources
if show_sources:
    st.subheader(KK["top_sources"])
    st.caption(KK["top_sources_expl"])
    top_src = filtered["source"].dropna().value_counts().reset_index()
    top_src.columns = ["source","count"]
    if top_src.empty:
        st.write("Дереккөздер жоқ")
    else:
        fig = px.bar(top_src.head(30), x="count", y="source", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_src.head(100))

# Text length
if show_lengths:
    st.subheader(KK["text_length"])
    st.caption(KK["text_length_expl"])
    if filtered["text_len"].sum()==0:
        st.info(KK["no_text"])
    else:
        fig = px.histogram(filtered, x="text_len", nbins=40)
        st.plotly_chart(fig, use_container_width=True)

# WordCloud
if show_wordcloud:
    st.subheader(KK["wordcloud"])
    texts = filtered["text"].dropna().astype(str).tolist()
    if not texts:
        st.info(KK["no_text"])
    else:
        big_text = " ".join(texts)
        clean = re.sub(r"http\S+|\n|\t"," ",big_text)
        wc = WordCloud(width=800, height=400, collocations=False, background_color="white").generate(clean)
        fig_wc, ax = plt.subplots(figsize=(12,5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)