import json
import pickle
import copy
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

from .datasets import register


@register('ImgDataset')
class ImgDataset(Dataset):
    def __init__(self, anno_file,
                 split="train",
                 first_k=None,
                 test_mode=False,
                 data_ratio=None,
                 transform=None):
        
        super().__init__()
        self.anno_file = anno_file

        self.split = split
        self.test_mode = test_mode
        self.img_infos = self.load_annotations()
        
        if first_k:
            self.img_infos = self.img_infos[:first_k]
        if data_ratio is not None:
            assert data_ratio > 0 and data_ratio <= 1
            data_len = int(len(self.img_infos) * data_ratio)
            self.img_infos = random.sample(self.img_infos, data_len)
            
        self.transform = transform
        if self.transform is None:
            if not test_mode:
                self.transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ])
    
    def load_json_annotations(self):
        with open(self.anno_file, 'r') as f:
            data = json.load(f)

        if self.split:
            data = data[self.split]
        return data
    
    def load_annotations(self):
        assert self.anno_file.endswith('.json')
        return self.load_json_annotations()
    
    def prepare_train_imgs(self, idx):
        img_info = self.img_infos[idx]
        img_path = img_info["img_path"]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = img_info["label"]
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label, img_info
    
    def prepare_test_imgs(self, idx):
        img_info = self.img_infos[idx]
        img_path = img_info["img_path"]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = img_info["label"]
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label, img_info

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_imgs(idx)
        return self.prepare_train_imgs(idx)
    
if __name__ == "__main__":
    path = "/home/rz/code/Depth-Face-MAE/data/lock3dface.json"
    dataset = ImgDataset(path, split="train")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    import ipdb; ipdb.set_trace()