import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class Mapillary_Vistas_Segmentation_Data(Dataset):
    def __init__(self, path, image_transform=None, label_transform=None):
        self.image_transform = image_transform
        self.label_transform = label_transform        
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

        if self.image_transform:
                image = self.image_transform(image)
        else:
                image = transforms.ToTensor()(image)

        if self.label_transform:
                label = self.label_transform(label)
        else:
                label = transforms.ToTensor()(label)

        label = label.squeeze(0)  
        label = (label * 255).long()    

        return image, label