import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_samples(model, device, num_samples=10, diffusion_steps=100, image_size=32):
    """
    生成样本：从随机噪声开始，通过反向扩散过程生成图像
    """
    model.eval()
    
    # 创建随机噪声 - 初始样本（不需要梯度）
    samples = torch.randn(num_samples, 3, image_size, image_size, device=device)
    
    print(f"开始反向扩散过程，共{diffusion_steps}步...")
    
    # 反向扩散过程
    for step in range(diffusion_steps):
        # 当前时间步（从1到0）
        current_time = 1.0 - step / diffusion_steps
        t = torch.full((num_samples,), current_time, device=device)
        
        # 关键修复：创建需要梯度的副本，而不是修改原始张量
        samples_with_grad = samples.clone().requires_grad_(True)
        
        # 预测噪声
        predicted_noise = model(samples_with_grad, t)
        
        # 手动计算梯度：目标是让预测的噪声最小化
        noise_loss = torch.mean(predicted_noise ** 2)
        
        # 计算梯度（只对samples_with_grad计算）
        gradients = torch.autograd.grad(
            outputs=noise_loss,
            inputs=samples_with_grad,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # 应用梯度下降步骤到原始samples
        learning_rate = 0.1 / diffusion_steps
        samples = samples - learning_rate * gradients.detach()
        
        # 添加少量噪声（除了最后一步）
        if step < diffusion_steps - 1:
            noise_scale = 0.2 * (1 - current_time)
            samples = samples + noise_scale * torch.randn_like(samples)
        
        # 数值稳定性：裁剪值
        samples = torch.clamp(samples, -3.0, 3.0)
        
        if step % 20 == 0 or step == diffusion_steps - 1:
            print(f"步骤 {step}/{diffusion_steps}, 时间: {current_time:.3f}")
    
    # 转换为图像格式 [0, 1] 范围
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    
    # 绘制结果
    plot_samples(samples, num_samples)
    
    return samples.detach()

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