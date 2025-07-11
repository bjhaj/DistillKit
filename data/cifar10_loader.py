import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

def get_cifar10_transforms(train=True):
    """Get CIFAR-10 transforms with data augmentation for training."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])

def get_cifar10_loaders(batch_size=128, num_workers=2, pin_memory=True):
    """
    Get CIFAR-10 train and test data loaders.
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Whether to pin memory in GPU training
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    transform_train = get_cifar10_transforms(train=True)
    transform_test = get_cifar10_transforms(train=False)
    
    # Training set
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # Test set
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    testloader = DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes