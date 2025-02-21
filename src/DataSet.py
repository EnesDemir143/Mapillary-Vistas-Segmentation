import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class Mapillary_Vistas_Segmentation_Data(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.path = path
        self.image_path = os.path.join(self.path, 'images')
        self.label_path = os.path.join(self.path, 'labels')
        self.image_files = sorted(os.listdir(self.image_path))
        self.label_files = sorted(os.listdir(self.label_path))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, self.image_files[idx]))
        label = Image.open(os.path.join(self.label_path, self.label_files[idx]))

        if self.transform:

            image = self.transform(image)
            label = self.transform(label)
            
        else:
            
            image = transforms.ToTensor(image)
            label = transforms.ToTensor(label)

            return image, label