from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp import NLPAnalyzer
import uvicorn
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],       
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "project.json"

analyzer = NLPAnalyzer()
analyzer.train_models_from_file(str(DATA_FILE))

class AnalyzeRequest(BaseModel):
    text: str
    channel: str
    date: str

@app.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    message_obj = {
        "text": req.text,
        "channel": req.channel,
        "date": req.date
    }
    result = analyzer.analyze_single_message(message_obj)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)