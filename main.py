import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import atexit

# Flask
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS

# FastAPI
from fastapi import FastAPI, Query
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware as FastAPICORS
from pydantic import BaseModel
import uvicorn

# Project modules
from dotenv import load_dotenv
from neo4j import GraphDatabase
from nlp import NLPAnalyzer
from translations import get_translation

# Загрузка переменных окружения
load_dotenv()

# ==========================================
# КОНФИГУРАЦИЯ И ПУТИ
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "db.sqlite"
THESAURUS_FILE = DATA_DIR / "thesaurus.json"

# Создаем папку data если нет
DATA_DIR.mkdir(exist_ok=True)

# ==========================================
# БАЗА ДАННЫХ (SQLite)
# ==========================================
def get_db():
    """Получить подключение к БД"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Инициализация таблиц"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Таблица сообщений
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            date TEXT,
            text TEXT,
            io_type TEXT,
            emo_eval TEXT,
            fake_claim TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Таблица результатов анализа
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER,
            ner_entities TEXT,
            thesaurus_matches TEXT,
            llm_summary TEXT,
            sentiment_score REAL,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    ''')
    
    # Создаем индексы
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_source ON messages(source)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date)')
    
    conn.commit()
    conn.close()
    print("✅ База данных инициализирована с индексами")

init_db()

# ==========================================
# NLP ANALYZER
# ==========================================
analyzer = NLPAnalyzer()
try:
    if DB_PATH.exists():
        if hasattr(analyzer, 'train_models_from_db'):
            analyzer.train_models_from_db(str(DB_PATH))
        else:
            print("⚠️ Метод train_models_from_db не найден в nlp.py, пропускаем обучение.")
    print("✅ NLP моделі жүктелді")
except Exception as e:
    print(f"⚠️ NLP моделін жүктеу қатесі: {e}")

# ==========================================
# NEO4J И ТЕЗАУРУС
# ==========================================
driver = None
try:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    if uri and user and password:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("✅ Neo4j базасы қосылды")
    else:
        print("⚠️ Neo4j деректері .env файлында жоқ")
except Exception as e:
    print(f"⚠️ Neo4j қосылмады: {e}")
    driver = None

if driver:
    atexit.register(lambda: driver.close())

def load_thesaurus():
    """Загрузить тезаурус из JSON"""
    try:
        if THESAURUS_FILE.exists():
            with open(THESAURUS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Тезаурус жүктеу қатесі: {e}")
    return []

# ==========================================
# FLASK APP (Frontend UI)
# ==========================================
flask_app = Flask(__name__)
flask_app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")
CORS(flask_app)

@flask_app.context_processor
def inject_translations():
    lang = session.get('lang', 'kk')
    return {'t': get_translation(lang)}

# --- Flask Routes ---

@flask_app.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    lang = data.get('lang', 'kk')
    session['lang'] = lang
    return jsonify({'status': 'ok', 'lang': lang})

@flask_app.route('/get_language')
def get_language():
    lang = session.get('lang', 'kk')
    return jsonify({'lang': lang})

@flask_app.route('/')
def index():
    if 'lang' not in session:
        session['lang'] = 'kk'
    if 'role' in session:
        role = session['role']
        if role == 'admin': return redirect(url_for('admin_page'))
        elif role == 'analyst': return redirect(url_for('analytics_page'))
        elif role == 'linguist': return redirect(url_for('thesaurus_page'))
    return render_template('index.html')

@flask_app.route('/login', methods=['GET', 'POST'])
def login():
    if 'lang' not in session: session['lang'] = 'kk'
    if request.method == 'POST':
        role = request.form.get('role')
        password = request.form.get('password')
        valid_passwords = {
            'admin': os.getenv('ADMIN_PASSWORD', 'admin123'),
            'analyst': os.getenv('ANALYST_PASSWORD', 'analyst123'),
            'linguist': os.getenv('LINGUIST_PASSWORD', 'linguist123')
        }
        if role in valid_passwords and password == valid_passwords[role]:
            session['role'] = role
            session['logged_in'] = True
            if role == 'admin': return redirect(url_for('admin_page'))
            elif role == 'analyst': return redirect(url_for('analytics_page'))
            elif role == 'linguist': return redirect(url_for('thesaurus_page'))
        else:
            return render_template('login.html', error='Құпия сөз қате')
    return render_template('login.html')

@flask_app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@flask_app.route('/admin')
def admin_page():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM messages ORDER BY created_at DESC LIMIT 10')
    rows = cursor.fetchall()
    conn.close()
    
    data = [dict(row) for row in rows]
    return render_template('admin.html', data=data)

@flask_app.route("/analyze", methods=["POST"])
def analyze_json():
    """Анализ текста и сохранение в SQLite"""
    if request.is_json:
        req_data = request.get_json()
        text = req_data.get("text")
        channel = req_data.get("channel")
        date = req_data.get("date")
    else:
        text = request.form.get("text")
        channel = request.form.get("channel")
        date = request.form.get("date")

    if not text:
        return jsonify({"error": "Мәтін енгізілмеген"}), 400

    if not date:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message_to_analyze = {
        "text": text,
        "channel": channel or "Manual Input",
        "date": date
    }

    report = analyzer.analyze_single_message(message_to_analyze)
    analysis_data = report.get("analysis_report", {})
    sentiment_data = analysis_data.get("general_sentiment", {})

    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (source, date, text, io_type, emo_eval, fake_claim)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            report.get("source_info", {}).get("channel"),
            report.get("source_info", {}).get("date"),
            report.get("original_text"),
            analysis_data.get("predicted_info_operation_type"),
            sentiment_data.get("label"),
            str(analysis_data.get("is_anomaly"))
        ))
        
        message_id = cursor.lastrowid
        
        cursor.execute('''
            INSERT INTO analysis_results (message_id, ner_entities, thesaurus_matches, llm_summary)
            VALUES (?, ?, ?, ?)
        ''', (
            message_id,
            json.dumps(analysis_data.get("named_entities_recognition", []), ensure_ascii=False),
            json.dumps(analysis_data.get("military_terms_analysis", []), ensure_ascii=False),
            json.dumps(analysis_data.get("llm_expert_summary", {}), ensure_ascii=False)
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Ошибка сохранения в БД: {e}")

    return jsonify(report)

@flask_app.route('/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM messages WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_page'))

@flask_app.route('/analytics')
def analytics_page():
    if 'role' not in session: return redirect(url_for('login'))
    return render_template('analytics.html')

@flask_app.route('/thesaurus')
def thesaurus_page():
    if 'role' not in session: return redirect(url_for('login'))
    
    thesaurus = load_thesaurus()
    all_terms = []
    for term in thesaurus:
        for lang in ['kk', 'ru', 'en']:
            t_name = term.get(f'TT_{lang}')
            if t_name: all_terms.append(f"{t_name} ({lang.upper()})")
            
    return render_template('thesaurus.html', all_terms=all_terms)

# 🆕 НОВЫЕ ЭНДПОИНТЫ ДЛЯ ЛИНГВИСТА
@flask_app.route('/thesaurus/search', methods=['GET'])
def thesaurus_search():
    """Поиск термина в тезаурусе"""
    term = request.args.get('term', '').strip()
    language = request.args.get('language', 'EN').upper()
    
    if not term:
        return jsonify({'error': 'Термин не указан'}), 400
    
    thesaurus = load_thesaurus()
    results = {}
    
    # Ищем термин во всех языках
    for lang_code in ['kk', 'ru', 'en']:
        key = f'TT_{lang_code}'
        for item in thesaurus:
            if item.get(key) and term.lower() in item.get(key, '').lower():
                lang_upper = lang_code.upper()
                results[lang_upper] = {
                    'term': item.get(key),
                    'scope_notes': [
                        item.get(f'SN_{lang_code}')
                    ],
                    'relations': {
                        'BROADER_TERM': [{'term': item.get(f'BT_{lang_code}'), 'language': lang_upper}] if item.get(f'BT_{lang_code}') else [],
                        'NARROWER_TERM': [{'term': item.get(f'NT_{lang_code}'), 'language': lang_upper}] if item.get(f'NT_{lang_code}') else [],
                        'RELATED_TERM': [{'term': item.get(f'RT_{lang_code}'), 'language': lang_upper}] if item.get(f'RT_{lang_code}') else [],
                        'USED_FOR': [{'term': item.get(f'UF_{lang_code}'), 'language': lang_upper}] if item.get(f'UF_{lang_code}') else [],
                        'PART_OF': [{'term': item.get(f'PT_{lang_code}'), 'language': lang_upper}] if item.get(f'PT_{lang_code}') else [],
                        'LANGUAGE_EQUIVALENT': [{'term': item.get(f'LE_{lang_code}'), 'language': lang_upper}] if item.get(f'LE_{lang_code}') else []
                    }
                }
                break
    
    if not results:
        return jsonify({'error': 'Термин не найден'}), 404
    
    return jsonify({'results': results})

@flask_app.route('/thesaurus/add', methods=['POST'])
def thesaurus_add():
    """Добавление нового термина"""
    term = request.form.get('term', '').strip()
    language = request.form.get('language', 'EN').upper()
    scope_note = request.form.get('scope_note', '').strip()
    
    if not term:
        return jsonify({'error': 'Термин не указан'}), 400
    
    # Загружаем текущий тезаурус
    thesaurus = load_thesaurus()
    
    # Находим максимальный ID
    max_id = max([item.get('id', 0) for item in thesaurus], default=0)
    
    # Создаем новую запись
    new_term = {
        'id': max_id + 1,
        'TT_kz': term if language == 'KZ' else None,
        'TT_ru': term if language == 'RU' else None,
        'TT_en': term if language == 'EN' else None,
        'SN_kz': scope_note if language == 'KZ' else None,
        'SN_ru': scope_note if language == 'RU' else None,
        'SN_en': scope_note if language == 'EN' else None,
        'BT_kz': None, 'BT_ru': None, 'BT_en': None,
        'NT_kz': None, 'NT_ru': None, 'NT_en': None,
        'RT_kz': None, 'RT_ru': None, 'RT_en': None,
        'UF_kz': None, 'UF_ru': None, 'UF_en': None,
        'PT_kz': None, 'PT_ru': None, 'PT_en': None,
        'LE_kz': None, 'LE_ru': None, 'LE_en': None
    }
    
    # Добавляем связь если указана
    relation_type = request.form.get('relation_type', '').strip()
    related_term = request.form.get('related_term', '').strip()
    
    if relation_type and related_term:
        lang_suffix = language.lower()
        if relation_type == 'BT':
            new_term[f'BT_{lang_suffix}'] = related_term
        elif relation_type == 'NT':
            new_term[f'NT_{lang_suffix}'] = related_term
        elif relation_type == 'RT':
            new_term[f'RT_{lang_suffix}'] = related_term
        elif relation_type == 'UF':
            new_term[f'UF_{lang_suffix}'] = related_term
        elif relation_type == 'PT':
            new_term[f'PT_{lang_suffix}'] = related_term
        elif relation_type == 'LE':
            new_term[f'LE_{lang_suffix}'] = related_term
    
    # Добавляем в список
    thesaurus.append(new_term)
    
    # Сохраняем обратно в файл
    try:
        with open(THESAURUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(thesaurus, f, ensure_ascii=False, indent=4)
        return jsonify({'success': f'Термин "{term}" успешно добавлен'})
    except Exception as e:
        return jsonify({'error': f'Ошибка сохранения: {str(e)}'}), 500

# ==========================================
# FASTAPI APP (Backend API)
# ==========================================
api = FastAPI(title="Info Operations API", version="1.0")

api.add_middleware(
    FastAPICORS,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/api/stats/summary")
def get_stats_summary():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM messages')
    total_messages = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM messages WHERE DATE(created_at) = DATE('now')")
    analyzed_today = cursor.fetchone()[0]
    conn.close()
    
    thesaurus = load_thesaurus()
    
    return {
        'total_messages': total_messages,
        'analyzed_today': analyzed_today,
        'total_terms': len(thesaurus)
    }

@api.get("/api/messages/paginated")
def get_messages_paginated(
    page: int = Query(1, ge=1, description="Номер страницы"),
    per_page: int = Query(10, ge=1, le=100, description="Записей на странице"),
    source: str = Query(None, description="Фильтр по источнику"),
    date_from: str = Query(None, description="Дата начала (YYYY-MM-DD)"),
    date_to: str = Query(None, description="Дата окончания (YYYY-MM-DD)")
):
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM messages WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM messages WHERE 1=1"
    params = []
    
    if source:
        query += " AND source = ?"
        count_query += " AND source = ?"
        params.append(source)
    
    if date_from:
        query += " AND date >= ?"
        count_query += " AND date >= ?"
        params.append(date_from)
    
    if date_to:
        query += " AND date <= ?"
        count_query += " AND date <= ?"
        params.append(date_to)
    
    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]
    
    offset = (page - 1) * per_page
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([per_page, offset])
    
    cursor.execute(query, params)
    messages = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    total_pages = (total + per_page - 1) // per_page
    
    return {
        'messages': messages,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    }

@api.get("/api/messages")
def get_all_messages(source: str = None, date_from: str = None, date_to: str = None):
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM messages WHERE 1=1"
    params = []
    
    if source:
        query += " AND source = ?"
        params.append(source)
    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)
        
    query += " ORDER BY created_at DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

# ==========================================
# MOUNT & RUN
# ==========================================
api.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Сервер іске қосылуда...")
    print("="*50)
    print(f"📍 URL: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    uvicorn.run(api, host="127.0.0.1", port=5000, log_level="info")