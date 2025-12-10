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
from translations import get_translation  # ✅ ИМПОРТ!

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
# TRANSLATIONS CONTEXT PROCESSOR
# ==========================================
@flask_app.context_processor
def inject_translations():
    """Внедрение переводов во все шаблоны"""
    lang = session.get('lang', 'kk')  # По умолчанию казахский
    return {'t': get_translation(lang)}

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
# FLASK ROUTES - Language
# ==========================================
@flask_app.route('/set_language', methods=['POST'])
def set_language():
    """Установить язык"""
    data = request.get_json()
    lang = data.get('lang', 'kk')
    session['lang'] = lang
    print(f"✅ Язык изменен на: {lang}")  # Debug
    return jsonify({'status': 'ok', 'lang': lang})

@flask_app.route('/get_language')
def get_language():
    """Получить текущий язык"""
    lang = session.get('lang', 'kk')
    print(f"📖 Текущий язык: {lang}")  # Debug
    return jsonify({'lang': lang})

# ==========================================
# FLASK ROUTES - Authentication
# ==========================================
@flask_app.route('/')
def index():
    """Главная страница"""
    # Устанавливаем язык по умолчанию
    if 'lang' not in session:
        session['lang'] = 'kk'
    
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
    # Устанавливаем язык по умолчанию
    if 'lang' not in session:
        session['lang'] = 'kk'
    
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

# ... остальной код (thesaurus routes, FastAPI, etc)

# ==========================================
# FASTAPI APP (Backend API)
# ==========================================
api = FastAPI(title="Info Operations API", version="1.0")

# CORS для API
api.add_middleware(
    FastAPICORS,
    allow_origins=["*"],
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

# ==========================================
# FASTAPI ROUTES
# ==========================================
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
    print("="*50 + "\n")
    
    uvicorn.run(api, host="127.0.0.1", port=5000, log_level="info")