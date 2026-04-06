import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class OvarianDataset(Dataset):
    def __init__(self, images_dir, masks_dir, train_list):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        with open(train_list, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]
        
        self.color_map = {
            (0, 0, 0): 0, (0, 0, 64): 1, (0, 0, 128): 2,
            (0, 64, 0): 3, (0, 128, 0): 4, (64, 0, 0): 5,
            (64, 0, 64): 6, (64, 64, 0): 7, (64, 64, 64): 8
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = cv2.imread(os.path.join(self.images_dir, img_id + ".JPG"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        # Превращаем картинку в тензор [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Возвращаем пока только картинку для теста
        return img_tensor, torch.zeros(256, 256)