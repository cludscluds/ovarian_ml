import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class OvarianDataset(Dataset):
    def __init__(self, images_dir, masks_dir, train_list_path):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        with open(train_list_path, "r") as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Загрузка снимка УЗИ
        img = cv2.imread(os.path.join(self.images_dir, img_id + ".JPG"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        # Загрузка маски
        mask = cv2.imread(os.path.join(self.masks_dir, img_id + ".PNG"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        
        # Делаем маску бинарной (0 или 1)
        binary_mask = (mask > 0).astype(np.float32)
        
        # Превращаем в тензоры
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(binary_mask).float().unsqueeze(0)
        
        return img_tensor, mask_tensor