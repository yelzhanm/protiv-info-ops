#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Static HTML есеп генераторы — project.json негізінде.
Шығарылатын файл: report.html
"""

import json
from pathlib import Path
from collections import Counter
import re
import base64
import io
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------- Көмекші функциялар (Streamlit скрипттен алынған/жарықталанған) ----------
def normalize_records(raw):
    records = []
    if isinstance(raw, dict) and not isinstance(raw, list):
        if "data" in raw and isinstance(raw["data"], dict) and any(k in raw["data"] for k in ["text", "date", "source"]):
            records = [raw]
        else:
            records = list(raw.values())
    elif isinstance(raw, list):
        records = raw
    else:
        records = [raw]

    out_rows = []
    for rec in records:
        data = {}
        if isinstance(rec, dict):
            data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        source = data.get("source") or rec.get("source") or (rec.get("meta") or {}).get("source")
        date = data.get("date") or rec.get("date") or (rec.get("meta") or {}).get("date")
        text = data.get("text") or rec.get("text") or rec.get("content") or (data.get("body") if isinstance(data.get("body"), str) else None)
        annotations = rec.get("annotations") or data.get("annotations") or rec.get("labels") or []
        if isinstance(text, dict):
            text_candidate = None
            for key in ("kk", "kz", "kaz", "Қаз", "kk-KZ", "kz-KZ"):
                if key in text:
                    text_candidate = text[key]
                    break
            if not text_candidate:
                for key in ("ru", "ru-RU", "ru_kz"):
                    if key in text:
                        text_candidate = text[key]
                        break
            if not text_candidate:
                for v in text.values():
                    if isinstance(v, str) and v.strip():
                        text_candidate = v
                        break
            text = text_candidate or ""
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

# sentiment (simple lexicon) — қысқаша
POS_WORDS_EN = {"good","great","positive","benefit","helpful","success","correct","true"}
NEG_WORDS_EN = {"bad","fake","false","wrong","danger","threat","risk","misinformation","lie"}
POS_WORDS_RU = {"хорошо","положительный","польза","полезно","успех","верно","правильно"}
NEG_WORDS_RU = {"плохо","негативный","опасно","угроза","ложь","фейк","неправда","риски","манипуляция"}
POS_WORDS_KK = {"жақсы","оң","пайда","көмек","табысты","дұрыс"}
NEG_WORDS_KK = {"жаман","теріс","қауіп","қатер","жалған","фейк","қателік","манипуляция"}

def detect_language_safe(text):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        if not text or len(text.strip()) < 5:
            return "unknown"
        return detect(text)
    except Exception:
        return "unknown"

def sentiment_lexicon_score(text, lang_hint=None):
    if not isinstance(text, str) or not text.strip():
        return {"score":0, "label":"нейтралды"}
    txt = text.lower()
    pos = neg = 0
    if lang_hint and lang_hint.startswith("ru"):
        for w in POS_WORDS_RU: pos += txt.count(w)
        for w in NEG_WORDS_RU: neg += txt.count(w)
    elif lang_hint and lang_hint.startswith("kk"):
        for w in POS_WORDS_KK: pos += txt.count(w)
        for w in NEG_WORDS_KK: neg += txt.count(w)
    else:
        for w in POS_WORDS_EN: pos += txt.count(w)
        for w in NEG_WORDS_EN: neg += txt.count(w)
        for w in POS_WORDS_RU: pos += txt.count(w)
        for w in NEG_WORDS_RU: neg += txt.count(w)
        for w in POS_WORDS_KK: pos += txt.count(w)
        for w in NEG_WORDS_KK: neg += txt.count(w)
    score = pos - neg
    if score > 0:
        label = "позитивті"
    elif score < 0:
        label = "негативті"
    else:
        label = "нейтралды"
    return {"score": score, "label": label, "pos": pos, "neg": neg}

def top_ngrams(texts, ngram_range=(1,1), top_n=20):
    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception:
        return pd.DataFrame(columns=["ngram","count"])
    if not texts:
        return pd.DataFrame(columns=["ngram","count"])
    vec = CountVectorizer(ngram_range=ngram_range, min_df=1, max_features=10000)
    X = vec.fit_transform(texts)
    sums = X.sum(axis=0)
    items = [(word, int(sums[0, idx])) for word, idx in vec.vocabulary_.items()]
    df_ng = pd.DataFrame(sorted(items, key=lambda x: x[1], reverse=True), columns=["ngram","count"])
    return df_ng.head(top_n)

def extract_places_fallback(text):
    keywords = [
        "Украина","Россия","Киев","Одесса","Москва","Алматы","Нур-Султан","Китай","Пекин","Токио",
        "Қазақстан","Казахстан","США","Америка","Франция","Париж","Германия","Берлин","Турция","Стамбул"
    ]
    return [word for word in keywords if word.lower() in text.lower()]

import pycountry
def detect_country(text):
    countries = {c.name: c.alpha_2 for c in pycountry.countries}
    found = []
    for name in countries.keys():
        if name.lower() in text.lower():
            found.append(name)
    return ", ".join(found) if found else "Unknown"

# ---------- Main ----------
def main():
    p = Path("project.json")
    if not p.exists():
        print("project.json табылмады — сол каталогқа орналастырыңыз немесе аргумент ретінде жол беріңіз.")
        sys.exit(1)

    raw = json.loads(p.read_text(encoding="utf-8"))
    rows = normalize_records(raw)
    df = pd.DataFrame(rows)
    df["parsed_date"] = df["date"].apply(parse_date_safe)
    df["text_len"] = df["text"].astype(str).apply(len)

    # фильтрация жоқ — просто общий отчет
    filtered = df.copy()

    # Overview
    total = len(df)
    filtered_count = len(filtered)
    min_date = df["parsed_date"].min()
    max_date = df["parsed_date"].max()
    avg_len = int(df["text_len"].mean()) if total>0 else 0

    # Time series
    ts = filtered.copy()
    ts["day"] = ts["parsed_date"].dt.floor("d")
    ts_counts = ts.groupby("day").size().reset_index(name="count").dropna()

    # Top sources
    top_src = filtered["source"].dropna().value_counts().reset_index()
    top_src.columns = ["source","count"]

    # WordCloud
    texts = filtered["text"].dropna().astype(str).tolist()
    big_text = " ".join(texts)
    clean = re.sub(r"http\S+|\n|\t", " ", big_text)
    wc_img_b64 = ""
    if clean.strip():
        wc = WordCloud(width=800, height=400, collocations=False, background_color="white").generate(clean)
        buf = io.BytesIO()
        plt.figure(figsize=(12,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        wc_img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    toks = [t.lower() for t in re.findall(r"\b[А-Яа-яA-Za-z]{3,}\b", clean)]
    tok_counts = Counter(toks)
    top_tokens = pd.DataFrame(tok_counts.most_common(100), columns=["сөз","сан"])

    # Annotations summary (labels + choices)
    all_labels = []
    agg_choices = Counter()
    for _, row in filtered.iterrows():
        lbls, chs = extract_annotation_info(row.get("annotations") or [])
        all_labels.extend(lbls)
        for k,v in chs.items():
            agg_choices[k] += v
    lbl_counts = pd.Series(all_labels).value_counts().reset_index() if all_labels else pd.DataFrame(columns=["метка","сан"])
    if not lbl_counts.empty:
        lbl_counts.columns = ["метка","сан"]

    # Language detection summary (sample)
    df['detected_lang'] = df['text'].fillna("").astype(str).apply(lambda t: detect_language_safe(t[:200]))
    lang_counts = df['detected_lang'].value_counts().reset_index()
    lang_counts.columns = ['тіл','сан']

    # N-grams
    texts_sample = filtered['text'].fillna("").astype(str).tolist()
    uni = top_ngrams(texts_sample, ngram_range=(1,1), top_n=20)
    bi = top_ngrams(texts_sample, ngram_range=(2,2), top_n=20)
    tri = top_ngrams(texts_sample, ngram_range=(3,3), top_n=20)

    # Geo (fallback)
    places_counter = Counter()
    country_counter = Counter()
    for txt in filtered['text'].fillna("").astype(str):
        if not txt.strip(): continue
        found_places = extract_places_fallback(txt)
        for ppp in found_places:
            places_counter[ppp] += 1
        country = detect_country(txt)
        country_counter[country] += 1

    # Sentiment (lexicon)
    sent_results = filtered['text'].fillna("").astype(str).apply(lambda t: sentiment_lexicon_score(t[:2000], lang_hint=detect_language_safe(t[:200])))
    sent_df = pd.DataFrame(list(sent_results))
    sent_summary = sent_df['label'].value_counts().reset_index()
    sent_summary.columns = ['тон','сан']

    # --- Построим Plotly графики и экспортируем их в HTML divs ---
    figs_html = {}
    try:
        if not ts_counts.empty:
            fig_ts = px.line(ts_counts, x="day", y="count", title="Уақыт бойынша тренд (жазбалар саны күн бойынша)")
            fig_ts.update_layout(xaxis_title="Күн", yaxis_title="Жазбалар саны")
            figs_html['time_series'] = pio.to_html(fig_ts, full_html=False, include_plotlyjs='cdn')
        if not top_src.empty:
            fig_src = px.bar(top_src.head(30), x="count", y="source", orientation="h", title="Топ дереккөздер")
            fig_src.update_layout(xaxis_title="Жазбалар саны", yaxis_title="Көз")
            figs_html['top_sources'] = pio.to_html(fig_src, full_html=False, include_plotlyjs=False)
        if not tok_counts:
            pass
        # text length histogram
        if not filtered["text_len"].isna().all():
            fig_len = px.histogram(filtered, x="text_len", nbins=40, title="Мәтін ұзындығының таралуы")
            fig_len.update_layout(xaxis_title="Мәтін ұзындығы (символдар)", yaxis_title="Фреквенция")
            figs_html['text_length'] = pio.to_html(fig_len, full_html=False, include_plotlyjs=False)
        # choices pie
        if agg_choices:
            ch_df = pd.DataFrame(agg_choices.most_common(), columns=["таңдау","сан"])
            fig_ch = px.pie(ch_df.head(20), values="сан", names="таңдау", title="Choices таралымы")
            figs_html['choices'] = pio.to_html(fig_ch, full_html=False, include_plotlyjs=False)
        # sentiment pie
        if not sent_summary.empty:
            fig_sent = px.pie(sent_summary, values='сан', names='тон', title='Тондардың таралымы (лексикон)')
            figs_html['sentiment'] = pio.to_html(fig_sent, full_html=False, include_plotlyjs=False)
    except Exception as e:
        print("Графиктер құруда қате:", e)

    # --- Собираем HTML (вставляем таблицы и графики) ---
    html_parts = []
    html_parts.append("<!doctype html><html lang='kk'><head><meta charset='utf-8'><title>Деректер аналитикасы — есеп</title>")
    html_parts.append("""<style>
    body{font-family:Arial,Helvetica,sans-serif; margin:20px; max-width:1200px;}
    h1,h2,h3{color:#1f2937}
    .metrics{display:flex;gap:1rem;flex-wrap:wrap}
    .metric{background:#f3f4f6;padding:12px;border-radius:8px;min-width:160px}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #ddd;padding:8px;text-align:left}
    .section{margin-top:30px;margin-bottom:30px}
    </style></head><body>""")

    html_parts.append("<h1>Деректер аналитикасы — статикалық есеп</h1>")
    html_parts.append("<p>Шығарылған файл: <strong>project.json</strong></p>")

    # Overview
    html_parts.append("<div class='section'><h2>Жалпы шолу</h2>")
    html_parts.append("<div class='metrics'>")
    html_parts.append(f"<div class='metric'><b>Жалпы жазба саны</b><div>{total}</div></div>")
    html_parts.append(f"<div class='metric'><b>Сүзілген жазбалар</b><div>{filtered_count}</div></div>")
    html_parts.append(f"<div class='metric'><b>Күндер аралығы</b><div>{(min_date.date() if pd.notnull(min_date) else '—')} — {(max_date.date() if pd.notnull(max_date) else '—')}</div></div>")
    html_parts.append(f"<div class='metric'><b>Орташа мәтін ұзындығы (символдар)</b><div>{avg_len}</div></div>")
    html_parts.append("</div></div>")

    # Time series
    html_parts.append("<div class='section'><h2>Уақыт бойынша тренд</h2>")
    if 'time_series' in figs_html:
        html_parts.append(figs_html['time_series'])
    else:
        html_parts.append("<p>Күндік деректер жоқ.</p>")
    html_parts.append("</div>")

    # Top sources
    html_parts.append("<div class='section'><h2>Топ дереккөздер</h2>")
    if 'top_sources' in figs_html:
        html_parts.append(figs_html['top_sources'])
    if not top_src.empty:
        html_parts.append(top_src.head(100).to_html(index=False))
    else:
        html_parts.append("<p>Дереккөздер көрсетілмеген.</p>")
    html_parts.append("</div>")

    # Text length
    html_parts.append("<div class='section'><h2>Мәтін ұзындығының таралуы</h2>")
    if 'text_length' in figs_html:
        html_parts.append(figs_html['text_length'])
    else:
        html_parts.append("<p>Мәтін ұзындығы дерегі жоқ.</p>")
    html_parts.append("</div>")

    # WordCloud + top tokens
    html_parts.append("<div class='section'><h2>WordCloud және топ сөздер</h2>")
    if wc_img_b64:
        html_parts.append(f"<img src='data:image/png;base64,{wc_img_b64}' alt='wordcloud' style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px'/>")
    else:
        html_parts.append("<p>WordCloud жасауға мәтін табылмады.</p>")
    if not top_tokens.empty:
        html_parts.append("<h3>Топ сөздер</h3>")
        html_parts.append(top_tokens.head(200).to_html(index=False))
    html_parts.append("</div>")

    # Annotations
    html_parts.append("<div class='section'><h2>Аннотациялар (labels және choices)</h2>")
    if not lbl_counts.empty:
        html_parts.append("<h3>Топ меткалар (labels)</h3>")
        html_parts.append(lbl_counts.to_html(index=False))
    else:
        html_parts.append("<p>Метка табылмады.</p>")
    if agg_choices:
        ch_df = pd.DataFrame(agg_choices.most_common(), columns=["таңдау","сан"])
        html_parts.append("<h3>Choices таралымы</h3>")
        html_parts.append(pio.to_html(px.pie(ch_df.head(20), values="сан", names="таңдау", title="Choices таралымы"), full_html=False, include_plotlyjs=False))
        html_parts.append(ch_df.to_html(index=False))
    html_parts.append("</div>")

    # Language, n-grams, geo, sentiment
    html_parts.append("<div class='section'><h2>Қосымша аналитика: Тіл, N-grams, Гео, Sentiment</h2>")
    html_parts.append("<h3>Тілдердің таралымы</h3>")
    html_parts.append(lang_counts.to_html(index=False))
    if 'sentiment' in figs_html:
        html_parts.append("<h3>Тондардың таралымы (лексикон)</h3>")
        html_parts.append(figs_html['sentiment'])
        html_parts.append(sent_summary.to_html(index=False))
    html_parts.append("<h3>Топ unigram</h3>")
    html_parts.append(uni.to_html(index=False))
    html_parts.append("<h3>Топ bigram</h3>")
    html_parts.append(bi.to_html(index=False))
    html_parts.append("<h3>Топ trigram</h3>")
    html_parts.append(tri.to_html(index=False))
    if places_counter:
        places_df = pd.DataFrame(places_counter.most_common(20), columns=["Орын","Саны"])
        html_parts.append("<h3>Ең жиі кездесетін орындар</h3>")
        html_parts.append(places_df.to_html(index=False))
    if country_counter:
        countries_df = pd.DataFrame(country_counter.most_common(), columns=["Ел","Саны"])
        html_parts.append("<h3>Елдер бойынша таралуы</h3>")
        html_parts.append(countries_df.to_html(index=False))
    html_parts.append("</div>")

    # Raw sample (optionally first 20 rows)
    html_parts.append("<div class='section'><h2>Raw — бірінші 20 жазба (text қысқаша)</h2>")
    sample = filtered.head(20).copy()
    sample['text'] = sample['text'].astype(str).str.slice(0,300)
    html_parts.append(sample.to_html(index=False))
    html_parts.append("</div>")

    # footer
    html_parts.append("<hr/><p>Бұл файл статикалық есеп — интерактивті серверлік мүмкіндіктер (фильтрлер, жүктеу, батырмалар) қосылмаған.</p>")
    html_parts.append("</body></html>")

    out_html = "\n".join(html_parts)
    out_path = Path.cwd() / "report.html"
    out_path.write_text(out_html, encoding="utf-8")
    print("Жасалды:", out_path)

if __name__ == "__main__":
    main()