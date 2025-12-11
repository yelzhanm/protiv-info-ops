import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "data" / "project.json"
DB_FILE = BASE_DIR / "data" / "db.sqlite"

def migrate():
    if not JSON_FILE.exists():
        print("❌ project.json табылмады")
        return

    print("🔄 Миграция басталды...")
    
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("❌ JSON форматы қате")
            return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Кестені құру (егер жоқ болса)
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

    count = 0
    for item in data:
        # JSON құрылымына байланысты өрістерді алу
        # Егер Label Studio форматы болса, 'data' ішінен аламыз
        source = item.get('source') or item.get('data', {}).get('source', 'Unknown')
        date = item.get('date') or item.get('data', {}).get('date', '')
        text = item.get('text') or item.get('data', {}).get('text', '')
        io_type = item.get('io_type') 
        emo = item.get('emo_eval')
        fake = item.get('fake_claim')

        # Annotation нәтижелерінен алу (Label Studio форматы болса)
        if 'annotations' in item and item['annotations']:
            for res in item['annotations'][0].get('result', []):
                if res.get('from_name') == 'io_type':
                    io_type = res['value']['choices'][0]
                elif res.get('from_name') == 'emo_eval':
                    emo = res['value']['choices'][0]
                elif res.get('from_name') == 'fake_claim':
                    fake = res['value']['choices'][0]

        if text:
            cursor.execute('''
                INSERT INTO messages (source, date, text, io_type, emo_eval, fake_claim)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (source, date, text, io_type, emo, str(fake)))
            count += 1

    conn.commit()
    conn.close()
    print(f"✅ {count} жазба сәтті көшірілді!")

if __name__ == "__main__":
    migrate()