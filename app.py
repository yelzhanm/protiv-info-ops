# app.py (full file)
from flask import Flask, render_template, url_for, request, redirect, jsonify
import json
from datetime import datetime
import spacy
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from flask_cors import CORS
from neo4j import GraphDatabase
import atexit
import urllib.parse
from dotenv import load_dotenv
from pathlib import Path


from nlp import NLPAnalyzer

app = Flask(__name__)
CORS(app)

DATA_FILE = "project.json"


nlp = spacy.load("ru_core_news_sm")
analyzer = NLPAnalyzer()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_PATH = BASE_DIR / "data" / "project.json"

DATA_FILE = str(DATA_FILE_PATH)

try:
    analyzer.train_models_from_file(str(DATA_FILE_PATH))
except Exception as e:
    print("⚠ NLP модельдерін жүктеу/оқыту кезінде қате:", e)


load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
atexit.register(lambda: driver.close())


def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def classify_text(text):
    text_lower = text.lower()
    io_type, emo_eval, fake_claim = "DEMORALIZATION", "Neutral", "False"

    if any(w in text_lower for w in ["фейк", "ложь", "дезинформация"]):
        io_type, fake_claim = "DISINFORMATION", "True"
    elif any(w in text_lower for w in ["угроза", "атака", "удар", "бомба", "дрон", "ракета"]):
        io_type, emo_eval = "INTIMIDATION", "Negative"
    elif any(w in text_lower for w in ["ненависть", "убить", "уничтожить"]):
        io_type, emo_eval = "HATE_INCITEMENT", "Negative"
    elif any(w in text_lower for w in ["паника", "страх"]):
        io_type, emo_eval = "PANIC_CREATION", "Negative"

    return io_type, emo_eval, fake_claim

def annotate_text(text):
    doc = nlp(text)
    label_map = {
        "LOC": "GEO_LOC",
        "PER": "AUTHOR_INTENT",
        "ORG": "SOURCE",
        "GPE": "GEO_LOC",
        "FAC": "TARGET_ENTITY",
        "DATE": "TIME_REF",
        "EVENT": "MIL_TERM"
    }

    annotations = []
    for ent in doc.ents:
        if ent.label_ in label_map:
            annotations.append({
                "label": label_map[ent.label_],
                "start": ent.start_char,
                "end": ent.end_char
            })
    return annotations

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin", methods=["GET", "POST"])
def admin_page():
    data = load_data()

    if request.method == "POST":
        text = request.form.get("text")
        source = request.form.get("source", "Manual Input")

        if text:
            io_type, emo_eval, fake_claim = classify_text(text)
            annotations = annotate_text(text)

            new_entry = {
                "source": source,
                "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                "text": text,
                "annotations": annotations,
                "io_type": io_type,
                "emo_eval": emo_eval,
                "fake_claim": fake_claim
            }

            data.insert(0, new_entry)
            save_data(data)

        return redirect(url_for("admin_page"))

    show_analysis = False
    return render_template("admin.html", data=data, show_analysis=show_analysis)

@app.route("/analyze", methods=["POST"])
def analyze_json():
    req_data = request.get_json()
    text = req_data.get("text")
    channel = req_data.get("channel")
    date = req_data.get("date")

    if not text:
        return jsonify({"error": "Мәтін енгізілмеген"}), 400

    if not date:
        date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    message_to_analyze = {
        "text": text,
        "channel": channel or "Manual Input",
        "date": date
    }

    report = analyzer.analyze_single_message(message_to_analyze)

    try:
        data = load_data()
        analysis_result = {
            "source": report.get("channel", channel),
            "date": report.get("date", date),
            "text": report.get("text", text),

            "io_type": report.get("IO_TYPE")
                        or report.get("ioType")
                        or report.get("io_type"),

            "emo_eval": report.get("EMO_EVAL")
                        or report.get("emoEval")
                        or report.get("emotion")
                        or report.get("emo_eval"),

            "fake_claim": report.get("FAKE_CLAIM")
                          or report.get("fakeClaim")
                          or report.get("fake_claim")
                          or report.get("fake"),

            "cmt_sent": report.get("CMT_SENT")
                         or report.get("comment")
                         or report.get("cmtSent")
                         or report.get("cmt_sent"),

            "annotations": report.get("annotations")
                           or report.get("ner")
                           or []
        }

        data.insert(0, analysis_result)
        save_data(data)

    except Exception as e:
        print("❌ Талдауды сақтауда қате:", e)

    return jsonify(report)

@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    data = load_data()

    if 0 <= record_id < len(data):
        data.pop(record_id)
        save_data(data)

    return redirect(url_for("admin_page"))

@app.route("/analytics")
def analytics_page():
    analitika_path = Path("analitika_kk.py")
    if analitika_path.exists():
        try:
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(analitika_path)])
            webbrowser.open("http://localhost:8504")
            return "<h3>Аналитика ашылуда...</h3>"
        except Exception as e:
            return f"<h3>Қате: {e}</h3>"
    else:
        return "<h3>Файл analitika_kk.py табылған жоқ!</h3>"


# -------------------------------------------------
# ТЕЗАУРУС БӨЛІМІ (жаңартылған - екінші кодтың логикасы бойынша)
# -------------------------------------------------

def get_all_terms(language=None):
    """Get all terms from database (for dropdown)"""
    with driver.session() as session:
        if language:
            query = """
            MATCH (t:Term {language: $language})
            RETURN t.name as name
            ORDER BY t.name
            """
            result = session.run(query, language=language)
        else:
            query = """
            MATCH (t:Term)
            RETURN t.name as name, t.language as language
            ORDER BY t.language, t.name
            """
            result = session.run(query)
        
        terms = []
        for record in result:
            if 'language' in record and record['language']:
                terms.append(f"{record['name']} ({record['language']})")
            else:
                terms.append(record['name'])
        return terms

# -------------------------------------------------
# Тезаурус беті
# -------------------------------------------------
@app.route("/thesaurus")
def thesaurus_page():
    """Тезаурус интерфейсін көрсетеді"""
    all_terms = get_all_terms()
    return render_template("thesaurus.html", all_terms=all_terms)

# -------------------------------------------------
# Тезаурус: терминді іздеу (GET әдісі)
# -------------------------------------------------
@app.route("/thesaurus/search", methods=["GET"])
def thesaurus_search():
    """Search for a term in thesaurus (GET method)"""
    term = request.args.get('term', '').strip()
    language = request.args.get('language', 'EN')
    
    if not term:
        return jsonify({"error": "Please enter a search term"}), 400
    
    # Get the term with its relationships
    with driver.session() as session:
        query = """
        MATCH (t:Term {name: $term, language: $language})
        OPTIONAL MATCH (t)-[:BROADER_TERM]->(bt:Term)
        OPTIONAL MATCH (t)-[:NARROWER_TERM]->(nt:Term)
        OPTIONAL MATCH (t)-[:RELATED_TERM]->(rt:Term)
        OPTIONAL MATCH (t)-[:USED_FOR]->(uf:Term)
        OPTIONAL MATCH (t)-[:PART_OF]->(po:Term)
        OPTIONAL MATCH (t)-[:LANGUAGE_EQUIVALENT]->(le:Term)
        RETURN t.name as term,
               t.language as language,
               t.scope_notes as scope_notes,
               COLLECT(DISTINCT bt.name) as broader_terms,
               COLLECT(DISTINCT nt.name) as narrower_terms,
               COLLECT(DISTINCT rt.name) as related_terms,
               COLLECT(DISTINCT uf.name) as used_for,
               COLLECT(DISTINCT po.name) as part_of,
               COLLECT(DISTINCT le.name) as language_equivalents
        """
        
        result = session.run(query, term=term, language=language)
        record = result.single()
    
    if not record or not record['term']:
        return jsonify({"error": f'Term "{term}" not found in {language}'}), 404
    
    # Get equivalents in other languages
    equivalents = {}
    if record['language_equivalents']:
        with driver.session() as session:
            for equiv_term in record['language_equivalents']:
                if equiv_term:  # Check if not None
                    # Find which language this equivalent is in
                    lang_query = """
                    MATCH (t:Term {name: $term})
                    WHERE t.language IN ['EN', 'RU', 'KZ']
                    RETURN t.language as language
                    """
                    lang_result = session.run(lang_query, term=equiv_term)
                    lang_record = lang_result.single()
                    
                    if lang_record:
                        # Get the equivalent term's details
                        equiv_query = """
                        MATCH (t:Term {name: $term, language: $lang})
                        OPTIONAL MATCH (t)-[:BROADER_TERM]->(bt:Term)
                        OPTIONAL MATCH (t)-[:NARROWER_TERM]->(nt:Term)
                        OPTIONAL MATCH (t)-[:RELATED_TERM]->(rt:Term)
                        OPTIONAL MATCH (t)-[:USED_FOR]->(uf:Term)
                        OPTIONAL MATCH (t)-[:PART_OF]->(po:Term)
                        RETURN t.name as term,
                               t.language as language,
                               t.scope_notes as scope_notes,
                               COLLECT(DISTINCT bt.name) as broader_terms,
                               COLLECT(DISTINCT nt.name) as narrower_terms,
                               COLLECT(DISTINCT rt.name) as related_terms,
                               COLLECT(DISTINCT uf.name) as used_for,
                               COLLECT(DISTINCT po.name) as part_of
                        """
                        
                        equiv_result = session.run(equiv_query, term=equiv_term, lang=lang_record['language'])
                        equiv_record = equiv_result.single()
                        
                        if equiv_record:
                            lang = equiv_record['language']
                            equivalents[lang] = {
                                'term': equiv_record['term'],
                                'language': lang,
                                'scope_notes': equiv_record['scope_notes'] or [],
                                'relations': {
                                    'BROADER_TERM': [{'term': term, 'language': lang} for term in equiv_record['broader_terms'] if term],
                                    'NARROWER_TERM': [{'term': term, 'language': lang} for term in equiv_record['narrower_terms'] if term],
                                    'RELATED_TERM': [{'term': term, 'language': lang} for term in equiv_record['related_terms'] if term],
                                    'USED_FOR': [{'term': term, 'language': lang} for term in equiv_record['used_for'] if term],
                                    'PART_OF': [{'term': term, 'language': lang} for term in equiv_record['part_of'] if term],
                                    'LANGUAGE_EQUIVALENT': []
                                }
                            }
    
    # Add the main term to results
    results_by_language = {}
    
    # Main term
    main_lang = record['language']
    results_by_language[main_lang] = {
        'term': record['term'],
        'language': main_lang,
        'scope_notes': record['scope_notes'] or [],
        'relations': {
            'BROADER_TERM': [{'term': term, 'language': main_lang} for term in record['broader_terms'] if term],
            'NARROWER_TERM': [{'term': term, 'language': main_lang} for term in record['narrower_terms'] if term],
            'RELATED_TERM': [{'term': term, 'language': main_lang} for term in record['related_terms'] if term],
            'USED_FOR': [{'term': term, 'language': main_lang} for term in record['used_for'] if term],
            'PART_OF': [{'term': term, 'language': main_lang} for term in record['part_of'] if term],
            'LANGUAGE_EQUIVALENT': [{'term': term, 'language': '?'} for term in record['language_equivalents'] if term]
        }
    }
    
    # Add equivalents
    for lang, data in equivalents.items():
        results_by_language[lang] = data
    
    return jsonify({
        'search_term': term,
        'search_language': language,
        'results': results_by_language,
        'all_terms': get_all_terms()
    })

# -------------------------------------------------
# Тезаурус: терминді қосу (POST әдісі)
# -------------------------------------------------
@app.route("/thesaurus/add", methods=["POST"])
def thesaurus_add():
    """Add a new term with optional relationships (POST method)"""
    term = request.form.get('term', '').strip()
    language = request.form.get('language', 'EN')
    scope_note = request.form.get('scope_note', '').strip()
    
    # Get relationship data if provided
    related_term = request.form.get('related_term', '').strip()
    relation_type = request.form.get('relation_type', '')
    
    if not term:
        return jsonify({"error": "Term name is required"}), 400
    
    with driver.session() as session:
        # Create the term
        create_query = """
        MERGE (t:Term {name: $term, language: $language})
        ON CREATE SET t.created_at = timestamp()
        """
        
        if scope_note:
            create_query += "SET t.scope_notes = coalesce(t.scope_notes, []) + $scope_note"
        
        session.run(create_query, term=term, language=language, scope_note=[scope_note])
        
        # If a relationship is specified, create it
        if related_term and relation_type:
            # Parse the related term (format: "term (language)" or just "term")
            if '(' in related_term and ')' in related_term:
                # Extract term and language from "term (language)"
                parts = related_term.split('(')
                related_term_name = parts[0].strip()
                related_lang = parts[1].replace(')', '').strip()
            else:
                # Just term name, use same language
                related_term_name = related_term
                related_lang = language
            
            # Map relationship type to Neo4j relationship
            rel_mapping = {
                'BT': 'BROADER_TERM',
                'NT': 'NARROWER_TERM', 
                'RT': 'RELATED_TERM',
                'UF': 'USED_FOR',
                'PT': 'PART_OF',
                'LE': 'LANGUAGE_EQUIVALENT'
            }
            
            neo4j_rel = rel_mapping.get(relation_type, 'RELATES_TO')
            
            # Check if related term exists, create if not
            check_query = """
            MERGE (rt:Term {name: $related_term, language: $related_lang})
            """
            session.run(check_query, related_term=related_term_name, related_lang=related_lang)
            
            # Create the relationship
            if relation_type == 'BT':
                # BT: New term is narrower than related term (new term -> related term is BROADER)
                rel_query = f"""
                MATCH (t:Term {{name: $term, language: $lang}})
                MATCH (rt:Term {{name: $related_term, language: $related_lang}})
                MERGE (t)-[:{neo4j_rel}]->(rt)
                """
            elif relation_type == 'NT':
                # NT: New term is broader than related term (new term <- related term is NARROWER)
                # Actually NT means new term has narrower terms, so related term is narrower than new term
                # So: related term -> new term is NARROWER_TERM
                rel_query = f"""
                MATCH (t:Term {{name: $term, language: $lang}})
                MATCH (rt:Term {{name: $related_term, language: $related_lang}})
                MERGE (rt)-[:{neo4j_rel}]->(t)
                """
            elif relation_type == 'LE':
                # LE: Create bidirectional language equivalent
                rel_query = f"""
                MATCH (t:Term {{name: $term, language: $lang}})
                MATCH (rt:Term {{name: $related_term, language: $related_lang}})
                MERGE (t)-[:{neo4j_rel}]->(rt)
                MERGE (rt)-[:{neo4j_rel}]->(t)
                """
            else:
                # For RT, UF, PT: new term -> related term
                rel_query = f"""
                MATCH (t:Term {{name: $term, language: $lang}})
                MATCH (rt:Term {{name: $related_term, language: $related_lang}})
                MERGE (t)-[:{neo4j_rel}]->(rt)
                """
            
            session.run(rel_query, 
                       term=term, lang=language,
                       related_term=related_term_name, related_lang=related_lang)
    
    # Get all terms for dropdown
    all_terms = get_all_terms()
    
    return jsonify({
        "success": f'Term "{term}" added successfully in {language}',
        "all_terms": all_terms
    })

# -------------------------------------------------
# Тезаурус: API терминді іздеу (JSON үшін)
# -------------------------------------------------
@app.route("/search_term", methods=["GET"])
def search_term_api():
    """API for searching terms (for JSON response)"""
    term_name = request.args.get("term")
    language = request.args.get("language", "EN")

    if not term_name:
        return jsonify({"error": "Please provide a term"}), 400

    # Use the same logic as thesaurus_search but with different response format
    with driver.session() as session:
        query = """
        MATCH (t:Term {name: $term, language: $language})
        OPTIONAL MATCH (t)-[:BROADER_TERM]->(bt:Term)
        OPTIONAL MATCH (t)-[:NARROWER_TERM]->(nt:Term)
        OPTIONAL MATCH (t)-[:RELATED_TERM]->(rt:Term)
        OPTIONAL MATCH (t)-[:USED_FOR]->(uf:Term)
        OPTIONAL MATCH (t)-[:PART_OF]->(po:Term)
        OPTIONAL MATCH (t)-[:LANGUAGE_EQUIVALENT]->(le:Term)
        RETURN t.name as term,
               t.language as language,
               t.scope_notes as scope_notes,
               COLLECT(DISTINCT bt.name) as broader_terms,
               COLLECT(DISTINCT nt.name) as narrower_terms,
               COLLECT(DISTINCT rt.name) as related_terms,
               COLLECT(DISTINCT uf.name) as used_for,
               COLLECT(DISTINCT po.name) as part_of,
               COLLECT(DISTINCT le.name) as language_equivalents
        """
        
        result = session.run(query, term=term_name, language=language)
        record = result.single()

    if not record or not record['term']:
        return jsonify({"error": f"Term '{term_name}' not found in {language}"}), 404

    term_data = {
        "name": record['term'],
        "language": record['language'],
        "scope_notes": record['scope_notes'] or [],
        "broader_terms": record['broader_terms'],
        "narrower_terms": record['narrower_terms'],
        "related_terms": record['related_terms'],
        "used_for": record['used_for'],
        "part_of": record['part_of'],
        "language_equivalents": record['language_equivalents']
    }

    return jsonify(term_data)

# -------------------------------------------------
# Тезаурус: API терминді қосу (JSON үшін)
# -------------------------------------------------
@app.route("/add_term", methods=["POST"])
def add_term_api():
    """API for adding terms (for JSON response)"""
    data = request.get_json() if request.is_json else request.form.to_dict()

    term_name = data.get("term") or data.get("name")
    language = data.get("language", "EN")
    scope_note = data.get("scope_note", "")
    relation_type = data.get("relation_type", "")
    related_term = data.get("related_term", "")

    if not term_name:
        return jsonify({"error": "Term name is required"}), 400

    with driver.session() as session:
        # Create the term
        create_query = """
        MERGE (t:Term {name: $term, language: $language})
        ON CREATE SET t.created_at = timestamp()
        """
        
        if scope_note:
            create_query += "SET t.scope_notes = coalesce(t.scope_notes, []) + $scope_note"
        
        session.run(create_query, term=term_name, language=language, scope_note=[scope_note])
        
        # If a relationship is specified, create it
        if related_term and relation_type:
            # Parse the related term (format: "term (language)" or just "term")
            if '(' in related_term and ')' in related_term:
                parts = related_term.split('(')
                related_term_name = parts[0].strip()
                related_lang = parts[1].replace(')', '').strip()
            else:
                related_term_name = related_term
                related_lang = language
            
            # Map relationship type to Neo4j relationship
            rel_mapping = {
                'BT': 'BROADER_TERM',
                'NT': 'NARROWER_TERM', 
                'RT': 'RELATED_TERM',
                'UF': 'USED_FOR',
                'PT': 'PART_OF',
                'LE': 'LANGUAGE_EQUIVALENT'
            }
            
            neo4j_rel = rel_mapping.get(relation_type, 'RELATES_TO')
            
            # Check if related term exists, create if not
            check_query = """
            MERGE (rt:Term {name: $related_term, language: $related_lang})
            """
            session.run(check_query, related_term=related_term_name, related_lang=related_lang)
            
            # Create the relationship
            if relation_type == 'BT':
                rel_query = f"""
                MATCH (t:Term {{name: $term, language: $lang}})
                MATCH (rt:Term {{name: $related_term, language: $related_lang}})
                MERGE (t)-[:{neo4j_rel}]->(rt)
                """
            elif relation_type == 'NT':
                rel_query = f"""
                MATCH (t:Term {{name: $term, language: $lang}})
                MATCH (rt:Term {{name: $related_term, language: $related_lang}})
                MERGE (rt)-[:{neo4j_rel}]->(t)
                """
            elif relation_type == 'LE':
                rel_query = f"""
                MATCH (t:Term {{name: $term, language: $lang}})
                MATCH (rt:Term {{name: $related_term, language: $related_lang}})
                MERGE (t)-[:{neo4j_rel}]->(rt)
                MERGE (rt)-[:{neo4j_rel}]->(t)
                """
            else:
                rel_query = f"""
                MATCH (t:Term {{name: $term, language: $lang}})
                MATCH (rt:Term {{name: $related_term, language: $related_lang}})
                MERGE (t)-[:{neo4j_rel}]->(rt)
                """
            
            session.run(rel_query, 
                       term=term_name, lang=language,
                       related_term=related_term_name, related_lang=related_lang)
    
    return jsonify({
        "message": f"Term '{term_name}' added successfully in {language}",
        "term": term_name,
        "language": language
    })

# -------------------------------------------------
# Run server
# -------------------------------------------------
if __name__ == "__main__":
    # debug режимінде жергілікті түрде ашып көру ыңғайлы
    app.run(debug=True, port=5000)