from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: int
    classe: str
    confidence: float
    score_cancer: float
    interpretation: str