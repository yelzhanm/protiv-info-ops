import json
import warnings
import numpy as np
import sqlite3
import os
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline, logging
from datetime import datetime
from pathlib import Path
import joblib
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
THESAURUS_FILE = BASE_DIR / "data" / "thesaurus.json"
DB_PATH = BASE_DIR / "data" / "db.sqlite"

# 🆕 ФИКСИРОВАННЫЕ КАТЕГОРИИ IO_TYPE
VALID_IO_TYPES = [
    "DISINFORMATION",
    "DEMORALIZATION", 
    "DISCREDITATION",
    "INTIMIDATION",
    "HATE_INCITEMENT",
    "PANIC_CREATION",
    "PROVOCATION",
    "AUTHORITY_UNDERSCORE"
]

# 🆕 ПЕРЕКЛЮЧЕНИЕ МЕЖДУ OLLAMA И GROQ
USE_GROQ = os.getenv('USE_GROQ', 'true').lower() == 'true'
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

# Старая модель (если USE_GROQ=false)
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

class NLPAnalyzer:
    def __init__(self):
        print("--- NLP-анализатор инициализацияланды ---")
        self.thesaurus = self._load_thesaurus()
        self.io_classifier = None
        self.vectorizer = None
        self.anomaly_model = None
        self._load_hf_models()
        self.model_path = BASE_DIR / "data" / "models.pkl"
        
        # 🆕 Проверка LLM доступности
        if USE_GROQ:
            if not GROQ_API_KEY:
                print("⚠️ GROQ_API_KEY не найден в .env. LLM анализ будет недоступен.")
                print("Получите бесплатный ключ на: https://console.groq.com/keys")
            else:
                print("✅ Groq API настроен")
        else:
            print("⚠️ Используется Ollama (требует локального запуска)")

    def _load_thesaurus(self):
        try:
            with open(THESAURUS_FILE, "r", encoding="utf-8") as f:
                print(f"✅ Тезаурус '{THESAURUS_FILE}' сәтті жүктелді.")
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ ҚАТЕ: Тезаурус файлы '{THESAURUS_FILE}' табылмады.")
            return []

    def _load_hf_models(self):
        print("🔄 Hugging Face модельдері жүктелуде...")
        self.ner_model = pipeline(
            "ner", 
            model="Babelscape/wikineural-multilingual-ner", 
            aggregation_strategy="simple"
        )
        self.sentiment_model = pipeline(
            "text-classification", 
            model="blanchefort/rubert-base-cased-sentiment"
        )
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Hugging Face модельдері жүктелді.")

    def train_models_from_db(self, db_path=str(DB_PATH)):
        """Модельдерді SQLite базасынан оқыту"""
        if self.model_path.exists():
            print("📥 Сақталған модельдер жүктелуде...")
            try:
                models = joblib.load(self.model_path)
                self.vectorizer = models['vectorizer']
                self.io_classifier = models['classifier']
                self.anomaly_model = models['anomaly']
                print("✅ Модельдер жүктелді!")
                return
            except Exception as e:
                print(f"⚠️ Модельді жүктеу қатесі: {e}, қайта оқыту басталады...")

        print("\n--- Модельдерді базадан үйрету басталды ---")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT text, io_type FROM messages WHERE io_type IS NOT NULL AND text IS NOT NULL")
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"❌ Базадан оқу қатесі: {e}")
            return

        if not rows:
            print("⚠️ ЕСКЕРТУ: База бос немесе деректер жоқ.")
            return

        texts = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        
        # 🆕 НОРМАЛИЗАЦИЯ МЕТОК К ВАЛИДНЫМ КАТЕГОРИЯМ
        normalized_labels = []
        for label in labels:
            # Попытка найти похожую валидную категорию
            label_upper = label.upper()
            if label_upper in VALID_IO_TYPES:
                normalized_labels.append(label_upper)
            else:
                # Если не нашли точное совпадение, берем DISINFORMATION по умолчанию
                print(f"⚠️ Неизвестная метка '{label}' заменена на DISINFORMATION")
                normalized_labels.append("DISINFORMATION")
        
        labels = normalized_labels

        unique_labels = set(labels)
        if len(unique_labels) >= 2:
            self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
            X = self.vectorizer.fit_transform(texts)
            self.io_classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.io_classifier.fit(X, labels)
            print(f"✅ АО классификаторы дайын ({len(rows)} жазба).")
        else:
            print(f"⚠️ Классификатор үйретілмеді. Бір ғана класс бар: {unique_labels}")

        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_model.fit(self.embedder.encode(texts))
        print("✅ Аномалия моделі дайын.")

        if self.io_classifier:
            print("💾 Модельдер сақталуда...")
            joblib.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.io_classifier,
                'anomaly': self.anomaly_model
            }, self.model_path)

    def analyze_single_message(self, message_object):
        text = message_object.get("text", "")
        if not text: return None

        ner_results = self.ner_model(text)
        sentiment_result = self.sentiment_model(text)[0]

        custom_rules = {"РФ": "ORG", "России": "LOC", "ВСУ": "ORG", "Украине": "LOC", "США": "LOC"}
        ner_entities = [{"entity": e.get("entity_group", "UNKNOWN"), "word": e.get("word")} for e in ner_results]
        for entity in ner_entities:
            if entity["word"] in custom_rules:
                entity["entity"] = custom_rules[entity["word"]]

        thesaurus_matches = self._find_thesaurus_terms(text)

        io_prediction = "DISINFORMATION"  # 🆕 ДЕФОЛТНОЕ ЗНАЧЕНИЕ
        if self.io_classifier and self.vectorizer:
            try:
                predicted = self.io_classifier.predict(self.vectorizer.transform([text]))[0]
                # 🆕 ПРОВЕРКА НА ВАЛИДНОСТЬ
                if predicted.upper() in VALID_IO_TYPES:
                    io_prediction = predicted.upper()
                else:
                    print(f"⚠️ Модель предсказала невалидную категорию: {predicted}")
            except Exception as e:
                print(f"⚠️ Ошибка предсказания: {e}")

        is_anomaly = False
        if self.anomaly_model:
            is_anomaly = True if self.anomaly_model.predict(self.embedder.encode([text]))[0] == -1 else False
        
        # 🆕 ИСПОЛЬЗОВАНИЕ GROQ ИЛИ OLLAMA
        llm_analysis = self._get_llm_summary(text, ner_entities, thesaurus_matches)

        return {
            "source_info": {"channel": message_object.get("channel"), "date": message_object.get("date")},
            "original_text": text,
            "analysis_report": {
                "predicted_info_operation_type": io_prediction,
                "is_anomaly": is_anomaly,
                "general_sentiment": {"label": sentiment_result['label'], "score": round(sentiment_result['score'], 3)},
                "military_terms_analysis": thesaurus_matches,
                "named_entities_recognition": ner_entities,
                "llm_expert_summary": llm_analysis
            }
        }

    def _find_thesaurus_terms(self, text, threshold=85):
        matches = []
        if not self.thesaurus: return matches
        term_types = {
            "Негізгі термин": ["TT_kz", "TT_ru", "TT_en"],
            "Синоним": ["UF_kz", "UF_ru", "UF_en"],
            "Байланысты термин": ["RT_kz", "RT_ru", "RT_en"]
        }
        for term in self.thesaurus:
            found = False
            for type_name, keys in term_types.items():
                for key in keys:
                    alias = term.get(key)
                    if alias and fuzz.partial_ratio(alias.lower(), text.lower()) > threshold:
                        matches.append({
                            "id": term.get("id"),
                            "term_kz": term.get("TT_kz"),
                            "term_ru": term.get("TT_ru"),
                            "term_en": term.get("TT_en"),
                            "matched_alias": alias,
                            "match_type": type_name
                        })
                        found = True
                        break
                if found: break
        return matches

    def _get_llm_summary(self, text, ner, thesaurus):
        """
        🆕 УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ДЛЯ LLM АНАЛИЗА
        Автоматически выбирает между Groq и Ollama
        """
        
        prompt = f"""Проанализируй текст на русском языке:

Текст: "{text}"

Найденные сущности: {ner}
Военные термины: {thesaurus}

Задача:
1. Напиши краткую сводку (2-3 предложения) о содержании текста
2. Оцени уровень угрозы от 1 до 5:
   - 1-2: Информационный/нейтральный
   - 3: Потенциально манипулятивный
   - 4-5: Явная дезинформация/угроза

Ответь СТРОГО в JSON формате:
{{
  "summary": "краткая сводка",
  "threat_level": число от 1 до 5
}}"""

        if USE_GROQ:
            return self._call_groq_api(prompt)
        else:
            return self._call_ollama_api(prompt)

    def _call_groq_api(self, prompt):
        """
        🆕 ВЫЗОВ GROQ API
        Документация: https://console.groq.com/docs/quickstart
        """
        if not GROQ_API_KEY:
            return {
                "summary": "LLM недоступен: GROQ_API_KEY не настроен",
                "threat_level": -1
            }
        
        try:
            import requests
            
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {GROQ_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'llama-3.3-70b-versatile',  # Бесплатная модель
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'Ты эксперт по анализу информационных операций. Отвечай ТОЛЬКО в JSON формате.'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'temperature': 0.3,
                    'max_tokens': 500
                },
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"⚠️ Groq API ошибка: {response.status_code}")
                return {
                    "summary": f"Ошибка API: {response.status_code}",
                    "threat_level": -1
                }
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Очистка от markdown
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
            
        except requests.Timeout:
            return {
                "summary": "Превышено время ожидания API",
                "threat_level": -1
            }
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка парсинга JSON: {e}")
            return {
                "summary": "Ошибка обработки ответа LLM",
                "threat_level": -1
            }
        except Exception as e:
            print(f"⚠️ Groq API ошибка: {e}")
            return {
                "summary": f"Ошибка LLM: {str(e)}",
                "threat_level": -1
            }

    def _call_ollama_api(self, prompt):
        """
        СТАРЫЙ МЕТОД С OLLAMA (для локальной разработки)
        """
        try:
            import ollama
            
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'timeout': 30}
            )
            
            return json.loads(response['message']['content'])
        except ImportError:
            return {
                "summary": "Ollama не установлен. Используйте USE_GROQ=true",
                "threat_level": -1
            }
        except Exception as e:
            print(f"⚠ Ollama қатесі: {e}")
            return {
                "summary": f"Ошибка Ollama: {str(e)}",
                "threat_level": -1
            }

if __name__ == "__main__":
    analyzer = NLPAnalyzer()
    analyzer.train_models_from_db()