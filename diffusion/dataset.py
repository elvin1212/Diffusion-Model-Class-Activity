import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloader(batch_size=32, num_workers=2):
    """
    Get CIFAR-10 dataset DataLoader
    """
    # Define data preprocessing steps
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Convert data from [0,1] range to [-1,1] range, more suitable for diffusion model training
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Set domestic mirror sources to accelerate download
    torchvision.datasets.CIFAR10.mirrors = [
        "https://mirror.ntmm.de/pytorch/",
        "https://mirror.nju.edu.cn/pytorch/",
        "https://mirrors.tuna.tsinghua.edu.cn/pytorch/",
        "https://mirrors.ustc.edu.cn/pytorch/",
        "https://mirror.nju.edu.cn/pytorch/"
    ]
    
    # Load training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='../data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    
    # Load test dataset
    testset = torchvision.datasets.CIFAR10(
        root='../data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers
    )
    
    return trainloader, testloader

def denormalize(tensor):
    """
    Convert normalized tensor back to [0,1] range for display
    """
    return (tensor + 1) / 2