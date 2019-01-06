import torchvision.datasets as dsets
import torchvision.transforms as transforms
from data import PureImageFolder, IgnoreLabelDataset

def get_dataset(dataset_name, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset_name == 'cifar-10.train':
        dataset = dsets.CIFAR10(
            root='/share/data/cifar-10', train=True, download=True, transform=transform
        )
    elif dataset_name == 'cifar-10.valid':
        dataset = dsets.CIFAR10(
            root='/share/data/cifar-10', train=False, download=True, transform=transform
        )
    elif dataset_name =='imagenet.train':
        if img_size == 32:
            dataset = PureImageFolder(
                root='/share/data/imagenet/train_32x32', transform=transform
            )
        if img_size == 64:
            dataset = PureImageFolder(
                root='/share/data/imagenet/train_64x64', transform=transform
            )
    elif dataset_name =='imagenet.valid':
        if img_size == 32:
            dataset = PureImageFolder(
                root='/share/data/imagenet/valid_32x32', transform=transform
            )
        if img_size == 64:
            dataset = PureImageFolder(
                root='/share/data/imagenet/valid_64x64', transform=transform
            )
    elif dataset_name == 'celeba.train':
        dataset = PureImageFolder(
            root='/share/data/celeba', transform=transform
        )
    elif dataset_name == 'celeba.valid':
        dataset = PureImageFolder(
            root='/share/data/celeba', transform=transform
        )
        dataset = LimitedImageDataset(dataset, 50000)
    elif dataset_name == 'lsun-bed.train':
        dataset = dsets.LSUN(
            root='/share/data/lsun', classes=['bedroom_train'], transform=transforms
        )
    elif dataset_name == 'lsun-bed.valid':
        dataset = dsets.LSUN(
            root='/share/data/lsun', classes=['bedroom_val'], transform=transforms
        )
    else:
        raise Exception('Dataset name not found: ' + dataset_name)
    return IgnoreLabelDataset(dataset)