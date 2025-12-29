"""
Inference pipeline for new text.
"""

import joblib
from preprocess import preprocess

MODEL_PATH = "model/classifier.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict(text: str):
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    return model.predict(vec)[0]

if __name__ == "__main__":
    text = input("Введите текст: ")
    result = predict(text)
    print("Классификация:", result)
