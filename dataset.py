from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import pandas as pd
import numpy as np

import os
from typing import Optional
from pathlib import Path

class ChestXRayDataset(Dataset):
    def __init__(self, csv_file: Path, data_root: Path, transform: Optional[T.transforms] = None) -> None:
        super(ChestXRayDataset, self).__init__()
        
        self.df: pd.DataFrame = pd.read_csv(csv_file)
        self.image_root: Path = data_root / 'images'
        self.transform: T.transforms = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["Image Index"]
        image = Image.open(img_path).convert('L')
        label = row["target"]
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_pos_weight(self) -> float:
        num_pos = self.df["target"].sum()
        num_neg = len(self.df) - num_pos
        
        if num_pos > 0:
            return num_neg / num_pos
        else:
            return 0.0