# analitika_kk.py
"""
Streamlit қосымшасы — JSON-файлдан толық аналитика (барлығы қазақша).
Функциялар:
 - автоматты түрде project.json жүктеу
 - Файл жүктеу (file_uploader) немесе жергілікті жол көрсету
 - Обзор (саны, диапазон датасы, орташа ұзындық)
 - Уақыт бойынша тренд (қатар)
 - Топ дереккөздер (sources)
 - Мәтін ұзындығының таралуы
 - Аннотациялар: меткалар (labels) және choices
 - WordCloud және топ сөздер
 - Экспорт CSV
Барлық батырмалар, тақырыптар және қысқа түсініктемелер қазақша.
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

# ---------------------------
# Қазақша мәтіндер (қосымшада қолданылатын барлық жолдар)
# ---------------------------
KK = {
    "title": " Деректер аналитикасы ",
    "subtitle": "Толық деректер файлын, яғни project.json файлын жүктеп, графиктер мен статистикалар көру.",
    "upload_prompt": "JSON файлын таңдаңыз (немесе төмендегі жолды пайдаланыңыз):",
    "local_path_prompt": "Жергілікті файл жолын көрсету (опционалды):",
    "use_local_button": "Жергілікті файлды жүктеу",
    "no_file": "Файл табылған жоқ. Жоғарыдағы батырма арқылы JSON файлын таңдаңыз немесе төменге файл жолын енгізіңіз.",
    "overview": "Жалпы шолу",
    "total_records": "Жалпы жазба саны",
    "filtered_records": "Сүзгіленген жазбалар",
    "date_range": "Файлдағы күндер аралығы",
    "avg_len": "Орташа мәтін ұзындығы (символдар)",
    "time_series": "Уақыт бойынша тренд (жазбалар саны күн бойынша)",
    "time_series_expl": "Бұл график күн бойынша жазбалар санының динамикасын көрсетеді. Көрсетілген күндерде хабарламалар/жазбалар көбейген/азайғанын бақылай аламыз.",
    "top_sources": "Топ дереккөздер (source)",
    "top_sources_expl": "Ең көп жазба түскен дереккөздер (каналдар).",
    "text_length": "Мәтін ұзындығының таралуы",
    "text_length_expl": "Мәтіндер қаншалықты ұзын — таралу гистограммасы. Пайдаланады: шағын жазбалар/ұзын мақалалар анықтау үшін.",
    "annotations": "Аннотациялар — меткалар (labels) және таңдаулар (choices)",
    "annotations_expl": "Егер файлда аннотациялар болса, бұл бөлім меткалар жиілігі мен choice-тар таралуын көрсетеді.",
    "wordcloud": "WordCloud — жиі кездесетін сөздер",
    "wordcloud_expl": "Мәтіндерден алынған ең жиі кездесетін сөздер",
    "top_tokens": "Топ сөздер (төмендегі кесте)",
    "save_csv": "Сүзілген нәтижені CSV-ге сақтау",
    "csv_saved": "CSV сақталды:",
    "download_csv": "CSV жүктеу сілтемесі дайын",
    "no_text": "Мәтін өрістері бос — WordCloud және токендер шығарылмайды.",
    "select_filters": "Фильтрлер және параметрлер",
    "sources_filter": "Дереккөздерді таңдау (фильтр)",
    "date_filter": "Күн аралығын таңдаңыз",
    "load_error": "JSON оқу қателігі: ",
    "ready": "Қош келдіңіз, бұл аналитика бөлімі!",
    "uploader_instructions": "Жүктеу аяқталғаннан кейін, сайдбардан қажетті аналитикаларды қосып/өшіріңіз."
}

JSON_PATH = Path("project.json")
raw_data = None
if JSON_PATH.exists():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        st.success(f"{KK['ready']} (автоматты түрде {JSON_PATH} жүктелді)")
    except Exception as e:
        st.warning(KK["load_error"] + str(e))
        raw_data = None
# ---------------------------
# Көмекші функциялар
# ---------------------------
@st.cache_data
def safe_load_json_file(file_like):
    try:
        return json.load(file_like)
    except Exception as e:
        raise

def normalize_records(raw):
    """
    Қарапайым нормализация:
    - raw мүмкін тізім, dict-of-records немесе бір жазба.
    - Запистерден мүмкін болатын өрістерді тартады: data.source, data.date, data.text, annotations.
    """
    records = []
    # кейде raw — dict-of-records
    if isinstance(raw, dict) and not isinstance(raw, list):
        # егер dict ішінде 'data' және 'text' бар болса — treat as single record
        if "data" in raw and isinstance(raw["data"], dict) and any(k in raw["data"] for k in ["text", "date", "source"]):
            records = [raw]
        else:
            # dict of id->record
            records = list(raw.values())
    elif isinstance(raw, list):
        records = raw
    else:
        records = [raw]

    out_rows = []
    for rec in records:
        # попытка достать разные варианты местоположения полей
        data = {}
        if isinstance(rec, dict):
            data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        # поля
        source = data.get("source") or rec.get("source") or (rec.get("meta") or {}).get("source")
        date = data.get("date") or rec.get("date") or (rec.get("meta") or {}).get("date")
        # text может лежать в data.text, rec['text'], rec['content']
        text = data.get("text") or rec.get("text") or rec.get("content") or (data.get("body") if isinstance(data.get("body"), str) else None)
        annotations = rec.get("annotations") or data.get("annotations") or rec.get("labels") or []
        # если text — dict (на разных языках), попробуем выбрать қазақша('kk') или 'kz', болмаса 'ru'/'en' fallback
        if isinstance(text, dict):
            text_candidate = None
            for key in ("kk", "kz", "kaz", "Қаз", "kk-KZ", "kz-KZ"):
                if key in text:
                    text_candidate = text[key]
                    break
            if not text_candidate:
                for key in ("ru", "ru-RU", "ru_kz", "ru-RU"):
                    if key in text:
                        text_candidate = text[key]
                        break
            if not text_candidate:
                # берем произвольно первый текстовый элемент
                for v in text.values():
                    if isinstance(v, str) and v.strip():
                        text_candidate = v
                        break
            text = text_candidate or ""
        # итоговая строка
        out = {
            "source": source,
            "date": date,
            "text": text or "",
            "annotations": annotations
        }
        out_rows.append(out)
    return out_rows

def parse_date_safe(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return pd.NaT
    try:
        return pd.to_datetime(d)
    except Exception:
        try:
            s = str(d)
            s = s.replace("T", " ").replace("Z", "")
            return pd.to_datetime(s)
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
            r_type = r.get("type") or (r.get("value") and "labels" if isinstance(r.get("value"), dict) and ("labels" in r.get("value")) else None)
            if r_type == "labels" or (isinstance(r.get("value"), dict) and "labels" in r.get("value")):
                v = r.get("value") or {}
                if isinstance(v, dict) and "labels" in v:
                    for lab in v.get("labels", []):
                        labels.append(lab)
                elif "text" in v and isinstance(v["text"], str):
                    labels.append(v["text"])
            if r_type == "choices" or (isinstance(r.get("value"), dict) and "choices" in r.get("value")):
                v = r.get("value") or {}
                chs = v.get("choices") or []
                for ch in chs:
                    choices[ch] += 1
            if a.get("label"):
                labels.append(a.get("label"))
    return labels, choices

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=KK["title"], layout="wide")
st.title(KK["title"])
st.caption(KK["subtitle"])

# ---- Автоматты жолмен оқу (егер project.json бар болса) ----
JSON_PATH = Path("project.json")  # <-- егер файл осылай аталған болса, ол автоматты түрде оқылады
raw_data = None
if JSON_PATH.exists():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        st.success(f"{KK['ready']} (автоматты түрде {JSON_PATH} жүктелді)")
    except Exception as e:
        st.warning(KK["load_error"] + str(e))
        raw_data = None

# Файл жүктеу интерфейсі (егер әлі жүктелмеген болса)
if raw_data is None:
    uploaded = st.file_uploader(KK["upload_prompt"], type=["json"])
    local_path = st.text_input(KK["local_path_prompt"], value="")  # опционалды
    if uploaded is not None:
        try:
            raw_data = safe_load_json_file(uploaded)
            st.success(KK["ready"])
        except Exception as e:
            st.error(KK["load_error"] + str(e))
            st.stop()
    elif local_path:
        p = Path(local_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                st.success(KK["ready"])
            except Exception as e:
                st.error(KK["load_error"] + str(e))
                st.stop()
        else:
            st.warning(KK["no_file"])
            st.stop()
    else:
        st.info(KK["no_file"])
        st.stop()

# Нормализация және DataFrame жасау
rows = normalize_records(raw_data)
df = pd.DataFrame(rows)
# parse dates
df["parsed_date"] = df["date"].apply(parse_date_safe)
df["text_len"] = df["text"].astype(str).apply(len)

# Сайдбар фильтрлері
st.sidebar.header(KK["select_filters"])
sources_opts = sorted(df["source"].dropna().unique()) if not df["source"].isna().all() else []
src_filter = st.sidebar.multiselect(KK["sources_filter"], options=sources_opts, default=None)
# date picker
min_date = df["parsed_date"].min()
max_date = df["parsed_date"].max()
try:
    default_range = (min_date.date() if pd.notnull(min_date) else None, max_date.date() if pd.notnull(max_date) else None)
except Exception:
    default_range = None
date_range = st.sidebar.date_input(KK["date_filter"], value=default_range)

# аналитика чекбокстары
st.sidebar.markdown("---")
show_overview = st.sidebar.checkbox(KK["overview"], value=True)
show_time = st.sidebar.checkbox(KK["time_series"], value=True)
show_sources = st.sidebar.checkbox(KK["top_sources"], value=True)
show_lengths = st.sidebar.checkbox(KK["text_length"], value=True)
show_annotations = st.sidebar.checkbox(KK["annotations"], value=True)
show_wordcloud = st.sidebar.checkbox(KK["wordcloud"], value=True)

# фильтрларды қолдану
filtered = df.copy()
if src_filter:
    filtered = filtered[filtered["source"].isin(src_filter)]
try:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        if start_d:
            filtered = filtered[filtered["parsed_date"] >= pd.to_datetime(start_d)]
        if end_d:
            filtered = filtered[filtered["parsed_date"] <= pd.to_datetime(end_d) + pd.Timedelta(days=1)]
except Exception:
    pass