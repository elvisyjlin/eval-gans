import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from glob import glob
from PIL import Image


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions):
    images = []
    for fnames in sorted(glob(os.path.join(dir, '*'))):
        if has_file_allowed_extension(fnames, extensions):
            images.append(fnames)
    return images

def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')

default_transform = transforms.Compose([
    transforms.ToTensor(), 
])

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=default_transform, loader=default_loader, 
                 extensions=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
        self.transform = transform
        self.loader = loader
        self.imgs = make_dataset(root, extensions)
    
    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.imgs)