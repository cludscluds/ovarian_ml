import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from models.core import UNet

# --- 1. ТВОЯ ФУНКЦИЯ ОЧИСТКИ ---
def clean_mask(mask):
    # Если на вход пришел тензор или вероятности, бинаризируем
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255

    kernel = np.ones((5,5), np.uint8)
    # Закрываем дырки
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Убираем мелкий шум
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Оставляем только самый большой объект (опухоль)
        main_contour = max(contours, key=cv2.contourArea)
        clean_m = np.zeros_like(mask)
        cv2.drawContours(clean_m, [main_contour], -1, 255, -1)
        return clean_m
    return mask

# --- 2. НАСТРОЙКИ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_model.pth"
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

# !!! ВПИШИ СЮДА ID СНИМКА, КОТОРЫЙ ХОЧЕШЬ ПРОВЕРИТЬ (без расширения) !!!
TARGET_ID = "1331" 

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
model = UNet(n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 4. ОБРАБОТКА КОНКРЕТНОГО СНИМКА ---
img_path = os.path.join(IMG_DIR, TARGET_ID + ".JPG")
mask_path = os.path.join(MASK_DIR, TARGET_ID + ".PNG")

if not os.path.exists(img_path):
    print(f"Ошибка: Файл {img_path} не найден!")
else:
    # Читаем и готовим изображение
    img_raw = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (256, 256))
    
    # Превращаем в тензор (B, C, H, W) и нормируем
    input_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
        raw_pred = torch.sigmoid(output).cpu().numpy()[0][0]
        
        # Очистка
        cleaned_pred = clean_mask(raw_pred)

    # Загружаем маску врача для сравнения (если она есть)
    has_mask = os.path.exists(mask_path)
    if has_mask:
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, (256, 256))

    # --- 5. ОТРИСОВКА ---
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 4, 1)
    plt.title(f"Оригинал: {TARGET_ID}")
    plt.imshow(img_res)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Маска врача")
    if has_mask:
        plt.imshow(true_mask, cmap='gray')
    else:
        plt.text(0.5, 0.5, 'Нет файла маски', ha='center')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title(f"ИИ Сырая (max: {raw_pred.max():.2f})")
    plt.imshow(raw_pred > 0.5, cmap='magma')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("ИИ Очищенная")
    plt.imshow(cleaned_pred, cmap='magma')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print(f"Проверка завершена для снимка: {TARGET_ID}")