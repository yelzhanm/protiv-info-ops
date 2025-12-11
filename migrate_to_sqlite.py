import json
import sqlite3
from pathlib import Path

# Жолдарды баптау
# Файл қай жерде тұрғанына қарамастан дұрыс жолды табамыз
BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "data" / "project.json"
DB_FILE = BASE_DIR / "data" / "db.sqlite"

def fill_database():
    # 1. Тексерулер
    if not JSON_FILE.exists():
        print(f"❌ Қате: Файл {JSON_FILE} табылмады!")
        return
    
    # data папкасын құру (егер жоқ болса)
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 2. JSON жүктеу
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("❌ Қате: JSON форматы бұрыс.")
            return

    # 3. Базаға қосылу
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # ⚠️ МАҢЫЗДЫ: Кестелерді құру (егер жоқ болса)
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

    print(f"🔄 Жүктеу басталды: {len(data)} жазба...")
    
    added_count = 0
    for item in data:
        # Деректерді алу (әр түрлі форматтар үшін)
        source = item.get('source') or item.get('data', {}).get('source', 'Unknown')
        date = item.get('date') or item.get('data', {}).get('date', '')
        text = item.get('text') or item.get('data', {}).get('text', '')
        
        # Меткаларды алу
        io_type = item.get('io_type')
        emo_eval = item.get('emo_eval')
        fake_claim = item.get('fake_claim')

        # Егер түбірде метка жоқ болса, annotation ішінен іздейміз
        if not io_type and 'annotations' in item and item['annotations']:
            try:
                for res in item['annotations'][0].get('result', []):
                    from_name = res.get('from_name')
                    val = res.get('value', {}).get('choices', [''])[0]
                    if from_name == 'io_type': io_type = val
                    elif from_name == 'emo_eval': emo_eval = val
                    elif from_name == 'fake_claim': fake_claim = val
            except:
                pass

        if text:
            try:
                cursor.execute('''
                    INSERT INTO messages (source, date, text, io_type, emo_eval, fake_claim)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (source, date, text, io_type, emo_eval, str(fake_claim)))
                added_count += 1
            except sqlite3.OperationalError as e:
                print(f"⚠️ Жазу қатесі: {e}")

    conn.commit()
    
    # 4. Нәтижені тексеру
    cursor.execute("SELECT COUNT(*) FROM messages")
    total = cursor.fetchone()[0]
    conn.close()
    
    print(f"✅ Дайын! Қосылған жазбалар: {added_count}")
    print(f"📊 Базадағы жалпы жазба саны: {total}")

if __name__ == "__main__":
    fill_database()