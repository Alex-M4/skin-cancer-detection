import io
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_model(model_path: str = "model.keras") -> tf.keras.Model:
    """Charge le modèle keras."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Modèle chargé avec succès : {model_path}")
        return model
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du modèle : {e}")
        raise e


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Prétraite une image reçue en bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)

        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Shape invalide : {img_array.shape}, attendu : (224, 224, 3)")

        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
        img_array = preprocess_input(img_array)        # ✅ IMPORTANT pour ResNet50

        return img_array
    except Exception as e:
        print(f"⚠️ Erreur lors du prétraitement : {e}")
        raise e


def predict(model: tf.keras.Model, image_bytes: bytes) -> dict:
    """Fait la prédiction du cancer de la peau binaire."""
    try:
        X = preprocess_image(image_bytes)

        preds = model.predict(X, verbose=0)
        score = float(preds[0][0])   # probabilité de cancer
        prediction = 1 if score > 0.5 else 0

        classes = ["Bénin", "Cancer"]
        pred_class = classes[prediction]
        interpretation = "✅ Bénin" if prediction == 0 else "🚨 Cancer suspect"

        # confidence = confiance dans la classe prédite
        confidence = score if prediction == 1 else (1 - score)

        return {
            "prediction": prediction,
            "classe": pred_class,
            "confidence": round(confidence * 100, 1),
            "score_cancer": round(score * 100, 2),
            "interpretation": interpretation
        }

    except Exception as e:
        print(f"⚠️ Erreur lors de la prédiction : {e}")
        raise e
    

### -------- OLD version -------- WORKING ALSO
# import tensorflow as tf
# import numpy as np

# def load_model():
#     """Charge le modèle Keras"""
#     print("🔄 Chargement du modèle Keras...")
#     # model = tf.keras.models.load_model("model.keras")
#     model = tf.keras.models.load_model("model.h5")
#     print("✅ Modèle chargé !")
#     return model

# def predict(model, image):
#     """Fait la prédiction binaire"""
#     try:
#         X = np.array(image, dtype=np.float32)

#         if X.shape != (224, 224, 3):
#             return {"error": f"Shape invalide. Reçu: {X.shape}, attendu: (224, 224, 3)"}

#         X = X / 255.0
#         X = np.expand_dims(X, axis=0)
        
#         pred = model.predict(X, verbose=0)
#         score = float(pred[0][0])  # Probabilité brut (0..1)
#         prediction = 1 if score > 0.5 else 0  # Seuil 50%
        
#         return {
#             "prediction": prediction,
#             "confidence": score,
#             "score": score
#         }
#     except Exception as e:
#         return {"error": str(e)}