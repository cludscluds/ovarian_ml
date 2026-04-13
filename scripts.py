import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from models.core import UNet
from models.data_loaders import OvarianDataset

# --- НАСТРОЙКИ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Если будет ошибка "Out of memory", поставь 4
EPOCHS = 20
LR = 0.0001

# 1. Модель, Оптимизатор, Лосс
model = UNet(n_classes=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 2. Данные
train_ds = OvarianDataset(
    images_dir="data/images", 
    masks_dir="data/masks", 
    train_list_path="data/train.txt"
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

print(f"Запуск на {DEVICE}. Найдено снимков: {len(train_ds)}")

# 3. Цикл обучения
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    print(f"Эпоха {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

# 4. Сохранение
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), "models/best_model.pth")
    print("Готово! Модель сохранена в models/best_model.pth")