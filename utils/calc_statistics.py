from dataset import ChestXRayDataset
from pathlib import Path
from torchvision import transforms as T
import numpy as np


if __name__ == "__main__":
    
    ds = ChestXRayDataset(csv_file=Path('data/NIH_Chest_X_Rays/splits/train.csv'),
                          data_root=Path('/home/avis/data/kaggle/chest_x_ray'),
                          transform=T.ToTensor())
    sum = 0.0
    squared_sum = 0.0
    num_pixels = 0
    for image, _ in ds:
        sum += image.sum()
        squared_sum += (image ** 2).sum()
        num_pixels += image.numel()
        
    mean = sum / num_pixels
    std = np.sqrt(squared_sum / num_pixels - mean ** 2)
    print(f"Dataset Mean {mean}, STD {std}")
        
    