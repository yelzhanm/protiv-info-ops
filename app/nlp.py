import json
import warnings
import ollama
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline, logging
from datetime import datetime

# --- ГЛОБАЛДЫ ПАРАМЕТРЛЕР ---
THESAURUS_FILE = r"C:\Users\User\Desktop\protiv-info-ops\thesaurus.json"
TRAINING_DATA_FILE = r"C:\Users\User\Desktop\protiv-info-ops\project.json"
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
        # Hugging Face модельдерін және embedder-ді жүктеу
        self._load_hf_models()

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

    def train_models_from_file(self, data_file_path):
        labeled_data = self._parse_label_studio_data(data_file_path)
        if not labeled_data:
            print("⚠ ЕСКЕРТУ: Деректер табылмады, модельдер үйретілмеді.")
            return

        print("\n--- Модельдерді үйрету басталды ---")
        texts = [item['text'] for item in labeled_data]
        labels = [item['label'] for item in labeled_data]

        unique_labels = set(labels)
        if len(unique_labels) >= 2:
            self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
            X = self.vectorizer.fit_transform(texts)
            self.io_classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.io_classifier.fit(X, labels)
            print("✅ АО классификаторы дайын.")
        else:
            print(f"⚠ Классификатор үйретілмеді. Бір ғана класс бар: {unique_labels}")

        baseline_texts = [item['text'] for item in labeled_data if item['label'] not in ['дезинформация', 'провокация']]
        if not baseline_texts:
            baseline_texts = texts
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_model.fit(self.embedder.encode(baseline_texts))
        print("✅ Аномалия моделі дайын.")

    def analyze_single_message(self, message_object):
        text = message_object.get("text", "")
        if not text: return None

        ner_results = self.ner_model(text)
        sentiment_result = self.sentiment_model(text)[0]

        # Кастом ережелер
        custom_rules = {"РФ": "ORG", "России": "LOC", "ВСУ": "ORG", "Украине": "LOC", "США": "LOC"}
        ner_entities = [{"entity": e.get("entity_group", "UNKNOWN"), "word": e.get("word")} for e in ner_results]
        for entity in ner_entities:
            if entity["word"] in custom_rules:
                entity["entity"] = custom_rules[entity["word"]]

        thesaurus_matches = self._find_thesaurus_terms(text)

        if self.io_classifier and self.vectorizer:
            io_prediction = self.io_classifier.predict(self.vectorizer.transform([text]))[0]
        else:
            io_prediction = "Классификация мүмкін емес"

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

    def _parse_label_studio_data(self, data_file_path):
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except FileNotFoundError: return None
        if not file_content.strip(): return []

        raw_data = []
        try: raw_data = json.loads(file_content)
        except json.JSONDecodeError:
            try: raw_data = json.loads('[' + file_content.replace('}{', '},{') + ']')
            except json.JSONDecodeError: return None

        if len(raw_data) == 1 and isinstance(raw_data[0], list):
            raw_data = raw_data[0]

        parsed_data = []
        for task in raw_data:
            if not isinstance(task, dict): continue
            text = task.get("data", {}).get("text")
            label = "belgisiz"
            try:
                for result_item in task["annotations"][0]["result"]:
                    if result_item.get("from_name") == "io_type":
                        label = result_item["value"]["choices"][0]
                        break
            except (KeyError, IndexError, TypeError):
                pass
            if text: parsed_data.append({"text": text, "label": label})
        return parsed_data


# --- Интерактивті тек басты орыннан іске қосу ---
if __name__ == "__main__":
    analyzer = NLPAnalyzer()
    analyzer.train_models_from_file(TRAINING_DATA_FILE)
    print("✅ Жүйе дайын. Интерактивті режим басталды.")
    while True:
        user_text = input("\n>>> Мәтінді енгізіңіз: ")
        if user_text.lower() == "exit": break
        channel_name = input(">>> Канал: ")
        date_str = input(">>> Дата (Enter = қазір): ")
        if not date_str: date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = analyzer.analyze_single_message({"text": user_text, "channel": channel_name, "date": date_str})
        print(json.dumps(report, ensure_ascii=False, indent=4))