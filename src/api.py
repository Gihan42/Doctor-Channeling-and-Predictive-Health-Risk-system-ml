# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests

# Import your local predictors
from src.cancer_predictor import CancerPredictor
from src.heart_attack_prediction import HeartAttackPredictor

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini AI configuration
API_KEY = ""
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ----------------- Data Models -----------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    messages: List[Message]

class PatientData(BaseModel):
    Age: int
    Gender: int
    BMI: float
    Smoking: int
    GeneticRisk: int
    PhysicalActivity: float
    AlcoholIntake: float
    CancerHistory: int

class HeartPatient(BaseModel):
    age: int
    sex: int
    cp: int
    trtbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# ----------------- Initialize Predictors -----------------
try:
    cancer_predictor = CancerPredictor(
        data_path='src/cancer_dataset/The_Cancer_data_1500_V2.csv')
    heart_predictor = HeartAttackPredictor(
        data_path='src/heart_attack_dataset/heart.csv')
except Exception as e:
    print(f"Error initializing predictors: {e}")

# ----------------- Routes -----------------
@app.get("/")
def read_root():
    return {"status": "Cancer & Heart Attack Prediction API is running"}

@app.post("/predict/cancer")
async def predict_cancer_risk(patient: PatientData):
    try:
        input_features = {
            'Age': patient.Age,
            'Gender': patient.Gender,
            'BMI': patient.BMI,
            'Smoking': patient.Smoking,
            'GeneticRisk': patient.GeneticRisk,
            'PhysicalActivity': patient.PhysicalActivity,
            'AlcoholIntake': patient.AlcoholIntake,
            'CancerHistory': patient.CancerHistory
        }
        result = cancer_predictor.predict(input_features)
        return {
            "status": "success",
            "prediction": result['prediction'],
            "probability": result['probability'],
            "interpretation": result['interpretation']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict/heart-attack")
async def predict_heart_attack(patient: HeartPatient):
    try:
        input_features = {
            'ge': patient.age,
            'sex': patient.sex,
            'cp': patient.cp,
            'trtbps': patient.trtbps,
            'chol': patient.chol,
            'fbs': patient.fbs,
            'restecg': patient.restecg,
            'thalachh': patient.thalach,
            'exng': patient.exang,
            'oldpeak': patient.oldpeak,
            'slp': patient.slope,
            'caa': patient.ca,
            'thall': patient.thal
        }
        result = heart_predictor.predict(input_features)
        return {
            "status": "success",
            "prediction": result['prediction'],
            "probability": result['probability'],
            "interpretation": result['interpretation']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ----------------- Gemini Chat Endpoint -----------------
@app.post("/predict/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        user_message = chat_request.messages[-1].content if chat_request.messages else "Hello!"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": user_message}
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": API_KEY
        }

        try:
            res = requests.post(URL, headers=headers, json=payload, timeout=15)
            res.raise_for_status()
            data = res.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Gemini API request failed: {e}")

        # Safely parse
        bot_response = "Sorry, I couldn’t understand the response from Gemini."
        try:
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    bot_response = parts[0]["text"]
        except Exception as e:
            print("Parsing error:", e)

        messages = chat_request.messages + [Message(role="assistant", content=bot_response)]
        return ChatResponse(response=bot_response, messages=messages)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat endpoint error: {e}")


