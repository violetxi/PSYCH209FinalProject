import os
import pdb
import pickle
import numpy as np
import torch

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

META_PATH = 'data/meta.pkl'


class VisualLauguageBuilder(Dataset):
    def __init__(self, meta_path=META_PATH, vision_only=True, analysis=False):
        # analysis is set to True after model is trained and we are
        # evaluating what's learned
        self.analysis = analysis
        self.vision_only = vision_only
        self.metas = pickle.load(open(meta_path, 'rb'))
        self.image_paths = []
        self.verbal_descriptors = []
        self.build_preprocess()
        self.__get_image_language_descriptors()        
        
    def __get_image_language_descriptors(self):
        for class_ in self.metas:            
            image_paths = self.metas[class_]['image_paths']
            verbal_descriptor = torch.from_numpy(
                self.metas[class_]['verbal_descriptor'])
            for image_path in image_paths:
                self.image_paths.append(image_path)
                self.verbal_descriptors.append(verbal_descriptor)                

    def __len__(self):
        return len(self.image_paths)

    # preprocess follows implementation at https://github.com/leaderj1001/SimSiam/blob/main/preprocess.py
    def build_preprocess(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if self.analysis:
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            preprocess = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.7),
                transforms.RandomGrayscale(0.2),
                transforms.ToTensor(),
                normalize,
            ])
        if self.vision_only:
            self.preprocess1 = preprocess
            self.preprocess2 = preprocess
        else:
            self.preprocess = preprocess            
        
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.vision_only:
            img1 = self.preprocess1(img)
            img2 = self.preprocess2(img)
            return img1, img2
        else:
            img = self.preprocess(img)
            verbal_desc = self.verbal_descriptors[idx]
            return img, verbal_desc            
            
                
if __name__ == '__main__':
    builder = VisualLauguageBuilder(META_PATH, vision_only=True)
    loader = DataLoader(builder, batch_size=10, shuffle=True, num_workers=0)
    for idx, imgs in enumerate(loader):
        print(idx, imgs.size())
        
    builder = VisualLauguageBuilder(META_PATH, vision_only=False)
    loader = DataLoader(builder, batch_size=10, shuffle=True, num_workers=0)
    for idx, items in enumerate(loader):
        img, verbal_descriptor = items
        print(idx, img.shape, verbal_descriptor.shape)
