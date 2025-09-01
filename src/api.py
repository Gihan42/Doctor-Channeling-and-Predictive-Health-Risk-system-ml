# api.py
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.cancer_predictor import CancerPredictor
from src.heart_attack_prediction import HeartAttackPredictor
from huggingface_hub import InferenceClient
from typing import List

app = FastAPI()

# Allow CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = InferenceClient(
    api_key=""
)
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
    cp: int  #Chest Pain Type
    trtbps: int #Resting Blood Pressure in mmHg
    chol: int #Serum Cholesterol in mg/d
    fbs: int #Fasting Blood Sugar > 120 mg/dl
    restecg: int #Resting Electrocardiographic results
    thalach: int  #Maximum Heart Rate Achieved
    exang: int    # Exercise Induced Angina
    oldpeak: float # ST depression induced
    slope: int    # Slope of the peak exercise ST segment
    ca: int       # Number of Major Vessels Colored by Fluoroscopy
    thal: int     # Thalassemia Stress Test Result

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    messages: List[Message]
# Initialize predictors
try:
    cancer_predictor = CancerPredictor(
        data_path='src/cancer_dataset/The_Cancer_data_1500_V2.csv')
    heart_predictor = HeartAttackPredictor(
        data_path='src/heart_attack_dataset/heart.csv')
except Exception as e:
    print(f"Error initializing predictors: {e}")

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
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/heart-attack")
async def predict_heart_attack(patient: HeartPatient):
    try:
        input_features = {
            'ge': patient.age,  # Changed from age to ge
            'sex': patient.sex,
            'cp': patient.cp,
            'trtbps': patient.trtbps,
            'chol': patient.chol,
            'fbs': patient.fbs,
            'restecg': patient.restecg,
            'thalachh': patient.thalach,  # No change needed (already correct)
            'exng': patient.exang,  # No change needed
            'oldpeak': patient.oldpeak,
            'slp': patient.slope,  # No change needed
            'caa': patient.ca,  # No change needed
            'thall': patient.thal  # No change needed
        }
        result = heart_predictor.predict(input_features)
        return {
            "status": "success",
            "prediction": result['prediction'],
            "probability": result['probability'],
            "interpretation": result['interpretation']
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/predict/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # Convert Pydantic messages to dict format for HuggingFace
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]

        # Create the completion request
        stream = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=500,
            stream=True
        )

        # Accumulate the response
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        # Update messages with assistant's response
        messages.append({"role": "assistant", "content": full_response})

        return ChatResponse(
            response=full_response,
            messages=[Message(**msg) for msg in messages]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))