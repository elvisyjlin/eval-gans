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

class PureImageFolder(data.Dataset):
    def __init__(self, root, transform=default_transform, loader=default_loader, 
                 extensions=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
        self.transform = transform
        self.loader = loader
        self.imgs = make_dataset(root, extensions)
    
    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
    
    def __len__(self):
        return len(self.imgs)

class IgnoreLabelDataset(data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

class LimitedImageDataset(data.Dataset):
    def __init__(self, orig, length):
        self.orig = orig
        self.length = length
    
    def __getitem__(self, index):
        return self.orig[index]
    
    def __len__(self):
        return self.length

# https://github.com/mttk/STL10
def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images

class UnlabeledSTL10(data.Dataset):
    def __init__(self, root, transform=default_transform):
        self.transform = transform
        self.imgs = read_all_images(os.path.join(root, 'stl10_binary', 'unlabeled_X.bin'))
    
    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
    
    def __len__(self):
        return len(self.imgs)