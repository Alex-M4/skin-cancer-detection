from fastapi import FastAPI, File, UploadFile, HTTPException
from schema import PredictionResponse
from model import load_model, predict

app = FastAPI(
    title="HAM10000 API",
    description="API de prédiction cancer/bénin avec FastAPI",
    version="1.0.0"
)

model = load_model()


@app.get("/")
def home():
    return {"message": "Welcome to the Alyra HAM10000 FastAPI!"}


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "HAM10000 API OK"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_ham(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        image_bytes = await file.read()
        result = predict(model, image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
from fastapi.responses import HTMLResponse

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return """
<!doctype html>
<html>
<head>
    <title>HAM10000 Détection Cancer/Peau</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; }
        .result { padding: 20px; border-radius: 10px; margin: 20px 0; }
        .benin { background: #d4edda; color: #155724; }
        .cancer { background: #f8d7da; color: #721c24; }
        input[type=file] { width: 100%; padding: 10px; }
        button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🩺 Détection Cancer de la Peau</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br><br>
        <button type="submit">🔍 Analyser Image</button>
    </form>
    
    <div id="result"></div>

    <script>
        // Auto-rafraîchir pour voir résultat
        if (window.location.search.includes('file=')) {
            setTimeout(() => location.reload(), 1000);
        }
    </script>
</body>
</html>
"""