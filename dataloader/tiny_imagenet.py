import os
import torch
import torchvision.transforms as transforms
from .cutout import Cutout
from torchvision import datasets

def get_tiny_imagenet(
    batch_size,
    num_workers,
):
    transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    
    data_train = datasets.ImageFolder(os.path.join('.', 'data', 'tiny-imagenet-200', 'train'), transform_train)
    data_test = datasets.ImageFolder(os.path.join('.', 'data', 'tiny-imagenet-200', 'test'), transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    
    
    return train_dataloader, test_dataloader, 200