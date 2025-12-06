import torch
from tqdm import tqdm

def linear_diffusion_schedule(t, max_beta=0.02):
    """线性扩散计划 - 手动计算噪声和信号率"""
    beta = t * max_beta
    alpha = 1.0 - beta
    signal_rates = torch.sqrt(alpha)
    noise_rates = torch.sqrt(1.0 - alpha)
    
    return noise_rates, signal_rates

def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    训练扩散模型，重点展示手动梯度控制
    """
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (real_images, _) in enumerate(pbar):
            # 将真实图像移动到设备
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]
            
            # 1. 创建基础张量（不需要梯度）
            base_noise = torch.randn_like(real_images, device=device)
            base_t = torch.rand(batch_size, device=device)
            
            # 2. 创建需要梯度的副本（正确方法）
            noise = base_noise.clone().requires_grad_(True)
            t = base_t.clone().requires_grad_(True)
            
            # 3. 计算噪声和信号率
            noise_rates, signal_rates = linear_diffusion_schedule(t)
            
            # 4. 创建带噪声的图像
            noisy_images = signal_rates.view(-1, 1, 1, 1) * real_images + \
                          noise_rates.view(-1, 1, 1, 1) * noise
            
            # 5. 模型预测噪声
            predicted_noise = model(noisy_images, t)
            
            # 6. 计算损失
            loss = criterion(predicted_noise, noise)
            
            # 7. 手动梯度计算和更新
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    return model