"""
Pipeline: OpenCV image input -> preprocessing -> PyTorch inference -> visualization.
"""

import cv2
import torch
import numpy as np
from model.cnn import CNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/mnist_cnn.pt"
IMG_PATH = "data/input.jpg"   # любое изображение числа 0-9

# 1. Загрузка модели
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 2. Загрузка и обработка картинки OpenCV
image = cv2.imread(IMG_PATH)

"""
OpenCV preprocessing steps:
- grayscale
- resize to model size
- threshold / normalize
- add batch + channel dimension
"""

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28))
normalized = resized / 255.0

tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

# 3. Инференс
with torch.no_grad():
    outputs = model(tensor)
    pred_class = outputs.argmax(1).item()

# 4. Визуализация в OpenCV (подпись результата)
result = image.copy()
cv2.putText(result, f"Prediction: {pred_class}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

# 5. Показ результата
cv2.imshow("Input Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Predicted Class: {pred_class}")
