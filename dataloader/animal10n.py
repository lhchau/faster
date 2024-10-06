import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets

def get_animal10n(
    batch_size,
    num_workers
):
    transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
        ])
    
    data_train = datasets.ImageFolder(os.path.join('.', 'data', 'Animal-10N', 'training'), transform_train)
    data_test = datasets.ImageFolder(os.path.join('.', 'data', 'Animal-10N', 'testing'), transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, test_dataloader, 10