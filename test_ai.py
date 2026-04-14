import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from models.core import UNet

# --- 1. ФУНКЦИЯ ОЧИСТКИ (МАГИЯ) ---
def clean_mask(mask):
    # Убеждаемся, что маска в формате 0-255
    mask = (mask * 255).astype(np.uint8)
    
    # Создаем ядро (размер фильтра). 5х5 — оптимально для 256х256
    kernel = np.ones((5,5), np.uint8)
    
    # Сначала "закрываем" (убираем дырки внутри опухоли)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Потом "открываем" (стираем мелкие точки и ворсинки снаружи)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Оставляем только самый большой объект (чтобы убрать артефакты от букв)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        main_contour = max(contours, key=cv2.contourArea)
        clean_m = np.zeros_like(mask)
        cv2.drawContours(clean_m, [main_contour], -1, 255, -1)
        return clean_m
    return mask

# --- 2. ОСТАЛЬНОЙ КОД ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_model.pth"
VAL_LIST = "data/val.txt"
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

model = UNet(n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with open(VAL_LIST, "r") as f:
    val_ids = [line.strip() for line in f.readlines() if line.strip()]

random_id = np.random.choice(val_ids)
print(f"Тестируем ИИ с очисткой: {random_id}")

img_raw = cv2.imread(os.path.join(IMG_DIR, random_id + ".JPG"))
img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
img_res = cv2.resize(img_rgb, (256, 256))
input_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0

with torch.no_grad():
    output = model(input_tensor.to(DEVICE))
    raw_pred = torch.sigmoid(output).cpu().numpy()[0][0]
    
    # Применяем очистку
    binary_pred = (raw_pred > 0.5).astype(np.uint8)
    cleaned_pred = clean_mask(binary_pred)

# Реальная маска врача
true_mask = cv2.imread(os.path.join(MASK_DIR, random_id + ".PNG"), cv2.IMREAD_GRAYSCALE)
true_mask = cv2.resize(true_mask, (256, 256))

# Отрисовка
plt.figure(figsize=(18, 5))
plt.subplot(1, 4, 1)
plt.title("ориг")
plt.imshow(img_res)

plt.subplot(1, 4, 2)
plt.title("маска врача")
plt.imshow(true_mask, cmap='gray')

plt.subplot(1, 4, 3)
plt.title("ии (сырая)")
plt.imshow(raw_pred > 0.5, cmap='magma')

plt.subplot(1, 4, 4)
plt.title("ии (очистка)")
plt.imshow(cleaned_pred, cmap='magma')

plt.show()