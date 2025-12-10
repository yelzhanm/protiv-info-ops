from flask import Flask, render_template, jsonify, request
from neo4j import GraphDatabase

# Flask қолданбасы
app = Flask(__name__)

# 🔗 Neo4j-пен байланыс
# ⚠ Өз логин/пароліңді қажет болса өзгерт
driver = GraphDatabase.driver("bolt://172.16.0.2:7687", auth=("neo4j", "iJQSUNd56KfY78w"))

# -----------------------
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