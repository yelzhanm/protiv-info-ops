import json
import warnings
import ollama
import numpy as np
import sqlite3  # SQLite қосылды
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline, logging
from datetime import datetime
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
THESAURUS_FILE = BASE_DIR / "data" / "thesaurus.json"
# TRAINING_DATA_FILE енді керек емес, DB_PATH қолданамыз
DB_PATH = BASE_DIR / "data" / "db.sqlite"

OLLAMA_MODEL = 'llama3'

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
            # Тек io_type бар жазбаларды аламыз (оқыту үшін)
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

        unique_labels = set(labels)
        if len(unique_labels) >= 2:
            self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
            X = self.vectorizer.fit_transform(texts)
            self.io_classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.io_classifier.fit(X, labels)
            print(f"✅ АО классификаторы дайын ({len(rows)} жазба).")
        else:
            print(f"⚠️ Классификатор үйретілмеді. Бір ғана класс бар: {unique_labels}")

        # Аномалия моделі (барлық мәтіндерге үйретеміз)
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

        io_prediction = "Белгісіз"
        if self.io_classifier and self.vectorizer:
            try:
                io_prediction = self.io_classifier.predict(self.vectorizer.transform([text]))[0]
            except:
                pass

        is_anomaly = False
        if self.anomaly_model:
            is_anomaly = True if self.anomaly_model.predict(self.embedder.encode([text]))[0] == -1 else False
            
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
        prompt = f"Текст: \"{text}\"\nСущности: {ner}\nТермины: {thesaurus}\nЗадача: Напиши краткую сводку (2-3 предложения) и оцени уровень угрозы (1-5). Ответ дай в JSON."
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'timeout': 120}
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"⚠ Ollama қатесі: {e}")
            return {"summary": "Ollama-дан жауап алынбады.", "threat_level": -1}

if __name__ == "__main__":
    analyzer = NLPAnalyzer()
    analyzer.train_models_from_db()