import torchvision.transforms.functional as F
import numpy as np
import random
import os
'''
dataset.py
功能：定义数据加载和预处理类。
主要作用：
1. 提供Dataset类（如FullDataset_new, TestDataset等），用于读取图像和标签数据。
2. 实现多种数据增强变换类（ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip等），用于在训练过程中增强数据的多样性。
'''
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}
    


class FullDataset_new(Dataset):
    def __init__(self, data_root, size, mode = 'train',valid_file = ''):
        super(FullDataset_new, self).__init__()
        if mode == 'train':
            self.image_root = os.path.join(data_root, 'train/images')
            self.gt_root = os.path.join(data_root, 'train/masks')
        else:
            self.image_root = os.path.join(data_root,'valid',valid_file, 'images')
            self.gt_root = os.path.join(data_root, 'valid',valid_file,'masks')

        self.images = [os.path.join(self.image_root,f) for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(self.gt_root,f) for f in os.listdir(self.gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.mode = mode

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                    Resize((size, size)),
                    ToTensor(),
                    Normalize()
            ])
            
    def __getitem__(self, idx):
            image = self.rgb_loader(self.images[idx])
            label = self.binary_loader(self.gts[idx])
            data = {'image': image, 'label': label}
            data = self.transform(data)
            mask_click  = data["label"].clone()
            _,pt_cup = self.random_click(np.array(mask_click).squeeze(0), point_label = 1)
            data["point"] = pt_cup
            return data
    
    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
    def random_click(self , mask, point_label = 1):
        max_label = max(set(mask.flatten()))
        # if round(max_label) == 0:
        if max_label.round() == 0:
            point_label = round(max_label)
        indices = np.argwhere(mask == max_label) 
        return point_label, indices[np.random.randint(len(indices))]         

class TestDataset(Dataset):
    def __init__(self, image_root, gt_root, size):
        self.images = [os.path.join(image_root,f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root,f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[idx])
        gt = np.array(gt)

        # name = self.images[idx].split('/')[-1]

        data = {'image': image, 'label': gt}
        return data
    
    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]
        mask_click  = gt.copy()
        _,pt_cup = self.random_click(np.array(mask_click), point_label = 1)
        self.index += 1
        return image, gt, name, pt_cup
    

    def random_click(self , mask, point_label = 1):
        max_label = max(set(mask.flatten()))
        # if round(max_label) == 0:
        if max_label.round() == 0:
            point_label = round(max_label)
        indices = np.argwhere(mask == max_label) 
        return point_label, indices[np.random.randint(len(indices))]    

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')