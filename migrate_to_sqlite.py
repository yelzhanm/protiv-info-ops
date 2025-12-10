import json
import sqlite3

with open('data/project.json') as f:
    data = json.load(f)

conn = sqlite3.connect('data/db.sqlite')
cursor = conn.cursor()

for item in data:
    cursor.execute('''
        INSERT INTO messages (source, date, text, io_type, emo_eval, fake_claim)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (item['source'], item['date'], item['text'], ...))

conn.commit()