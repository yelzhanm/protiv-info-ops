from flask import Flask, render_template, jsonify, request
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os


# Загружаем переменные окружения
load_dotenv()

app = Flask(__name__)

# Получаем данные из .env
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

if not password:
    print("⚠️ Внимание: Пароль для Neo4j не найден в .env!")

driver = GraphDatabase.driver(uri, auth=(user, password))
# 🔹 Басты бет (index.html)
# -----------------------
@app.route('/')
def index():
    return render_template('thesaurus.html')

# -----------------------
# 🔹 Барлық терминдерді шығару
# -----------------------
@app.route('/thesaurus')
def get_thesaurus():
    with driver.session() as session:
        result = session.run("MATCH (t:Term) RETURN t")
        data = [record["t"]._properties for record in result]
    return jsonify(data)

# -----------------------
# 🔹 Жаңа термин қосу (қажет болса POST арқылы)
# -----------------------
@app.route('/add', methods=['POST'])
def add_thesaurus():
    data = request.json
    with driver.session() as session:
        session.run("""
            CREATE (t:Term {
                id: $id,
                tt_kz: $tt_kz, tt_ru: $tt_ru, tt_en: $tt_en,
                sn_kz: $sn_kz, sn_ru: $sn_ru, sn_en: $sn_en,
                pt_kz: $pt_kz, pt_ru: $pt_ru, pt_en: $pt_en
            })
        """, data)
    return jsonify({"status": "success"})

# -----------------------
# 🔹 Іске қосу
# -----------------------
if __name__ == '__main__':
    app.run(debug=True)