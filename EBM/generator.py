import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import denormalize

def generate_samples(model, device, num_samples=10, steps=500, step_size=0.01, noise=0.005):
    """
    使用训练好的EBM模型生成样本
    """
    model.eval()
    
    # 从随机噪声开始
    samples = torch.randn(num_samples, 3, 32, 32).to(device) * 0.01
    
    print(f"开始Langevin Sampling过程，共{steps}步...")
    
    # Langevin Sampling过程
    for step in range(steps):
        samples.requires_grad_(True)
        energy = model(samples)
        grad = torch.autograd.grad(energy.sum(), samples)[0]
        
        # 更新样本
        samples = samples - step_size * grad + noise * torch.randn_like(samples)
        
        # 裁剪值域
        samples = torch.clamp(samples, -1.0, 1.0)
        
        # 不需要梯度进行下一步计算
        samples = samples.detach()
        
        if step % 100 == 0 or step == steps - 1:
            print(f"步骤 {step}/{steps}")
    
    # 转换为图像格式 [0, 1] 范围
    samples = denormalize(samples)
    
    # 绘制结果
    plot_samples(samples, num_samples)
    
    return samples

def plot_samples(samples, num_samples):
    """绘制生成的样本"""
    fig, axes = plt.subplots(1, min(num_samples, 10), figsize=(15, 2))
    
    if num_samples == 1:
        axes = [axes]
    elif num_samples > 10:
        axes = axes[:10]
    
    for i, ax in enumerate(axes):
        img = samples[i].cpu().permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
        ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.show()