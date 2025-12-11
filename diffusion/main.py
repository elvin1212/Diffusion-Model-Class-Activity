import torch
import torch.nn as nn
from model import get_model
from trainer import train_diffusion
from generator import generate_samples
from dataset import get_cifar10_dataloader
import os

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = get_model("Diffusion", image_size=32, num_channels=3)
    model.to(device)
    
    # 获取CIFAR-10数据加载器
    print("加载CIFAR-10数据集...")
    train_loader, test_loader = get_cifar10_dataloader(batch_size=64, num_workers=2)
    
    # 准备训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 训练模型
    print("开始训练扩散模型...")
    trained_model = train_diffusion(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        device=device, 
        epochs=20
    )
    
    # 生成样本
    print("生成样本...")
    samples = generate_samples(
        trained_model, 
        device=device, 
        num_samples=6, 
        diffusion_steps=50,
        image_size=32
    )
    
    # 保存模型
    torch.save(trained_model.state_dict(), "diffusion_model.pth")
    print("模型已保存为 diffusion_model.pth")
    
    print("程序执行完成！")

if __name__ == "__main__":
    main()