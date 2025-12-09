from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp import NLPAnalyzer
import uvicorn

# -------------------------
#   🌐 FastAPI App
# -------------------------
app = FastAPI()

# ---------------------------------------
#   🔥 CORS — Браузерден келетін OPTIONS
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # кез келген сайтқа рұқсат
    allow_credentials=True,
    allow_methods=["*"],          # POST, GET, OPTIONS, бәрі ашық
    allow_headers=["*"],
)

# ---------------------------------------
#   🤖 NLP Анализаторды жүктеу
# ---------------------------------------
analyzer = NLPAnalyzer()
analyzer.train_models_from_file(
    r"C:\Users\User\Desktop\protiv-info-ops\project.json"
)

# ---------------------------------------
#   📩 Request моделі
# ---------------------------------------
class AnalyzeRequest(BaseModel):
    text: str
    channel: str
    date: str

# ---------------------------------------
#   🚀 Негізгі API маршруты
# ---------------------------------------
@app.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    message_obj = {
        "text": req.text,
        "channel": req.channel,
        "date": req.date
    }
    result = analyzer.analyze_single_message(message_obj)
    return result

# ---------------------------------------
#   ▶ API серверін іске қосу
# ---------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)