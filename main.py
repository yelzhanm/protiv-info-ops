#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Объединенный сервер - Flask (UI) + FastAPI (API)
Запуск: python main.py
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from neo4j import GraphDatabase
import atexit

from fastapi import FastAPI, HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware as FastAPICORS
from pydantic import BaseModel
import uvicorn

from nlp import NLPAnalyzer
from translations import get_translation

# Загрузка переменных окружения
load_dotenv()

# Пути
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "db.sqlite"
THESAURUS_FILE = DATA_DIR / "thesaurus.json"

# Создаем папку data если нет
DATA_DIR.mkdir(exist_ok=True)

# ==========================================
# FLASK APP (Frontend)
# ==========================================
flask_app = Flask(__name__)
flask_app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")
CORS(flask_app)

# ==========================================
# DATABASE SETUP
# ==========================================
def init_db():
    """Инициализация SQLite базы данных"""
    conn = sqlite3.connect(DB_PATH)
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
    
    conn.commit()
    conn.close()
    print("✅ База данных инициализирована")

# Инициализируем БД при старте
init_db()

# ==========================================
# NEO4J CONNECTION
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

# ==========================================
# NLP ANALYZER
# ==========================================
analyzer = NLPAnalyzer()
try:
    # Пытаемся загрузить модели
    if DB_PATH.exists():
        analyzer.train_models_from_file(str(DB_PATH))
    print("✅ NLP моделі жүктелді")
except Exception as e:
    print(f"⚠️ NLP моделін жүктеу қатесі: {e}")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_db():
    """Получить подключение к БД"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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
# FLASK CONTEXT PROCESSOR
# ==========================================
@flask_app.context_processor
def inject_translations():
    """Внедрение переводов во все шаблоны"""
    lang = session.get('lang', 'kk')
    return {'t': get_translation(lang)}

# ==========================================
# FLASK ROUTES - Authentication
# ==========================================
@flask_app.route('/')
def index():
    """Главная страница"""
    # Если уже залогинен, редирект на соответствующую панель
    if 'role' in session:
        role = session['role']
        if role == 'admin':
            return redirect(url_for('admin_page'))
        elif role == 'analyst':
            return redirect(url_for('analytics_page'))
        elif role == 'linguist':
            return redirect(url_for('thesaurus_page'))
    
    return render_template('index.html')

@flask_app.route('/login', methods=['GET', 'POST'])
def login():
    """Страница входа"""
    if request.method == 'POST':
        role = request.form.get('role')
        password = request.form.get('password')
        
        # Проверка паролей из .env
        valid_passwords = {
            'admin': os.getenv('ADMIN_PASSWORD', 'admin123'),
            'analyst': os.getenv('ANALYST_PASSWORD', 'analyst123'),
            'linguist': os.getenv('LINGUIST_PASSWORD', 'linguist123')
        }
        
        if role in valid_passwords and password == valid_passwords[role]:
            session['role'] = role
            session['logged_in'] = True
            
            # Редирект на соответствующую страницу
            if role == 'admin':
                return redirect(url_for('admin_page'))
            elif role == 'analyst':
                return redirect(url_for('analytics_page'))
            elif role == 'linguist':
                return redirect(url_for('thesaurus_page'))
        else:
            return render_template('login.html', error='Құпия сөз қате')
    
    return render_template('login.html')

@flask_app.route('/logout')
def logout():
    """Выход"""
    session.clear()
    return redirect(url_for('index'))

# ==========================================
# FLASK ROUTES - Language
# ==========================================
@flask_app.route('/set_language', methods=['POST'])
def set_language():
    """Установить язык"""
    data = request.get_json()
    lang = data.get('lang', 'kk')
    session['lang'] = lang
    return jsonify({'status': 'ok'})

@flask_app.route('/get_language')
def get_language():
    """Получить текущий язык"""
    return jsonify({'lang': session.get('lang', 'kk')})

# ==========================================
# FLASK ROUTES - Admin Panel
# ==========================================
@flask_app.route('/admin')
def admin_page():
    """Админ панель"""
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    # Получаем сообщения из БД
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM messages ORDER BY created_at DESC LIMIT 100')
    messages = cursor.fetchall()
    conn.close()
    
    # Конвертируем в список словарей
    data = []
    for msg in messages:
        data.append({
            'id': msg['id'],
            'source': msg['source'],
            'date': msg['date'],
            'text': msg['text'],
            'io_type': msg['io_type'],
            'emo_eval': msg['emo_eval'],
            'fake_claim': msg['fake_claim']
        })
    
    return render_template('admin.html', data=data)

@flask_app.route('/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    """Удалить запись"""
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM messages WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    
    return redirect(url_for('admin_page'))

# ==========================================
# FLASK ROUTES - Analytics
# ==========================================
@flask_app.route('/analytics')
def analytics_page():
    """Страница аналитики"""
    if 'role' not in session:
        return redirect(url_for('login'))
    
    return render_template('analytics.html')

# ==========================================
# FLASK ROUTES - Thesaurus
# ==========================================
@flask_app.route('/thesaurus')
def thesaurus_page():
    """Тезаурус"""
    if 'role' not in session:
        return redirect(url_for('login'))
    
    # Загружаем все термины для datalist
    thesaurus = load_thesaurus()
    all_terms = []
    for term in thesaurus:
        for lang in ['kk', 'ru', 'en']:
            term_name = term.get(f'TT_{lang}')
            if term_name:
                all_terms.append(f"{term_name} ({lang.upper()})")
    
    return render_template('thesaurus.html', all_terms=all_terms)

@flask_app.route('/thesaurus/search', methods=['GET'])
def thesaurus_search():
    """Поиск термина в тезаурусе"""
    term = request.args.get('term', '').strip()
    language = request.args.get('language', 'EN').upper()
    
    if not term:
        return jsonify({"error": "Please enter a search term"}), 400
    
    thesaurus = load_thesaurus()
    
    # Поиск термина
    result = None
    for t in thesaurus:
        term_key = f'TT_{language.lower()}'
        if t.get(term_key, '').lower() == term.lower():
            result = t
            break
    
    if not result:
        return jsonify({"error": f"Term '{term}' not found in {language}"}), 404
    
    # Формируем ответ
    response = {
        'search_term': term,
        'search_language': language,
        'results': {
            language: {
                'term': result.get(f'TT_{language.lower()}'),
                'language': language,
                'scope_notes': [result.get(f'SN_{language.lower()}', '')],
                'relations': {
                    'BROADER_TERM': [{'term': result.get(f'BT_{language.lower()}'), 'language': language}],
                    'NARROWER_TERM': [{'term': result.get(f'NT_{language.lower()}'), 'language': language}],
                    'RELATED_TERM': [{'term': result.get(f'RT_{language.lower()}'), 'language': language}],
                    'USED_FOR': [{'term': result.get(f'UF_{language.lower()}'), 'language': language}],
                    'PART_OF': [{'term': result.get(f'PT_{language.lower()}'), 'language': language}],
                    'LANGUAGE_EQUIVALENT': []
                }
            }
        }
    }
    
    return jsonify(response)

@flask_app.route('/thesaurus/add', methods=['POST'])
def thesaurus_add():
    """Добавить термин в тезаурус"""
    term = request.form.get('term', '').strip()
    language = request.form.get('language', 'EN').upper()
    scope_note = request.form.get('scope_note', '').strip()
    
    if not term:
        return jsonify({"error": "Term name is required"}), 400
    
    # Загружаем текущий тезаурус
    thesaurus = load_thesaurus()
    
    # Создаем новый термин
    new_id = max([t.get('id', 0) for t in thesaurus], default=0) + 1
    new_term = {
        'id': new_id,
        f'TT_{language.lower()}': term,
        f'SN_{language.lower()}': scope_note
    }
    
    thesaurus.append(new_term)
    
    # Сохраняем обратно
    try:
        with open(THESAURUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(thesaurus, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "success": f"Term '{term}' added successfully in {language}",
            "term": term,
            "language": language
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# FASTAPI APP (Backend API)
# ==========================================
api = FastAPI(title="Info Operations API", version="1.0")

# CORS для API
api.add_middleware(
    FastAPICORS,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PYDANTIC MODELS
# ==========================================
class AnalyzeRequest(BaseModel):
    text: str
    channel: str
    date: str

class MessageResponse(BaseModel):
    id: int
    source: str
    date: str
    text: str
    io_type: str
    emo_eval: str
    fake_claim: str

# ==========================================
# FASTAPI ROUTES
# ==========================================
@api.post("/api/analyze")
def analyze_text(req: AnalyzeRequest):
    """Анализ текста через NLP"""
    if not req.text:
        raise HTTPException(status_code=400, detail="Мәтін енгізілмеген")
    
    message_obj = {
        "text": req.text,
        "channel": req.channel,
        "date": req.date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # NLP анализ
    try:
        report = analyzer.analyze_single_message(message_obj)
        
        # Сохраняем в БД
        conn = get_db()
        cursor = conn.cursor()
        
        analysis_data = report.get("analysis_report", {})
        sentiment_data = analysis_data.get("general_sentiment", {})
        
        cursor.execute('''
            INSERT INTO messages (source, date, text, io_type, emo_eval, fake_claim)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            req.channel,
            req.date,
            req.text,
            analysis_data.get("predicted_info_operation_type"),
            sentiment_data.get("label"),
            str(analysis_data.get("is_anomaly"))
        ))
        
        message_id = cursor.lastrowid
        
        # Сохраняем детальный анализ
        cursor.execute('''
            INSERT INTO analysis_results (message_id, ner_entities, thesaurus_matches, llm_summary, sentiment_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            message_id,
            json.dumps(analysis_data.get("named_entities_recognition", []), ensure_ascii=False),
            json.dumps(analysis_data.get("military_terms_analysis", []), ensure_ascii=False),
            json.dumps(analysis_data.get("llm_expert_summary", {}), ensure_ascii=False),
            sentiment_data.get("score", 0)
        ))
        
        conn.commit()
        conn.close()
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Талдау қатесі: {str(e)}")

@api.get("/api/stats/summary")
def get_stats_summary():
    """Получить краткую статистику"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Всего сообщений
    cursor.execute('SELECT COUNT(*) FROM messages')
    total_messages = cursor.fetchone()[0]
    
    # Сегодня проанализировано
    cursor.execute('''
        SELECT COUNT(*) FROM messages 
        WHERE DATE(created_at) = DATE('now')
    ''')
    analyzed_today = cursor.fetchone()[0]
    
    conn.close()
    
    # Термины из тезауруса
    thesaurus = load_thesaurus()
    
    return {
        'total_messages': total_messages,
        'analyzed_today': analyzed_today,
        'total_terms': len(thesaurus)
    }

@api.get("/api/messages")
def get_messages(limit: int = 100):
    """Получить список сообщений"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM messages ORDER BY created_at DESC LIMIT ?', (limit,))
    messages = cursor.fetchall()
    conn.close()
    
    result = []
    for msg in messages:
        result.append({
            'id': msg['id'],
            'source': msg['source'],
            'date': msg['date'],
            'text': msg['text'],
            'io_type': msg['io_type'],
            'emo_eval': msg['emo_eval'],
            'fake_claim': msg['fake_claim']
        })
    
    return result

# ==========================================
# MOUNT FLASK TO FASTAPI
# ==========================================
api.mount("/", WSGIMiddleware(flask_app))

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Сервер іске қосылуда...")
    print("="*50)
    print(f"📍 URL: http://127.0.0.1:5000")
    print(f"📊 Admin: http://127.0.0.1:5000/admin")
    print(f"📈 Analytics: http://127.0.0.1:5000/analytics")
    print(f"📚 Thesaurus: http://127.0.0.1:5000/thesaurus")
    print(f"🔧 API Docs: http://127.0.0.1:5000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(api, host="127.0.0.1", port=5000, log_level="info")