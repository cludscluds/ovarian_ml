import torch      # Подключаем нейросети
import cv2        # Подключаем работу с видео/фото
import numpy as np # Подключаем математику

# Проверяем, видит ли Python установленные модули
print("--- Проверка систем ---")
print(f"PyTorch готов к работе? {torch.__version__ is not None}")
print(f"OpenCV видит картинки? {cv2.__version__ is not None}")
print(f"Математика (Numpy) работает? {np.__version__ is not None}")

# Проверяем, есть ли у тебя видеокарта для ускорения (если нет - будет False, это норм)
print(f"Доступно ли ускорение через видеокарту (CUDA): {torch.cuda.is_available()}")