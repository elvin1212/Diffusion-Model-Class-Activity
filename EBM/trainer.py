import torch
from tqdm import tqdm

class Buffer:
    def __init__(self, device):
        self.device = device
        self.examples = None

    def sample_new_exmps(self, model, steps, step_size, noise_std, batch_size):
        # 从随机噪声开始
        fake_examples = torch.randn(batch_size, 3, 32, 32).to(self.device) * 0.01
        
        # Langevin Dynamics采样
        for _ in range(steps):
            fake_examples.requires_grad_(True)
            energy = model(fake_examples)
            grad = torch.autograd.grad(energy.sum(), fake_examples)[0]
            
            # 更新样本
            fake_examples = fake_examples - step_size * grad + \
                           noise_std * torch.randn_like(fake_examples)
            
            # 裁剪值域
            fake_examples = torch.clamp(fake_examples, -1.0, 1.0)
            
        return fake_examples.detach()

def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10, 
              alpha=0.1, steps=60, step_size=10, noise=0.005):
    """
    训练Energy-Based模型
    """
    model.to(device)
    model.train()
    
    # 创建样本缓冲区
    buffer = Buffer(device)
    
    for epoch in range(epochs):
        total_loss = 0
        total_cdiv_loss = 0  # 对比散度损失
        total_reg_loss = 0   # 正则化损失
        
        pbar = tqdm(data_loader, desc=f'EBM Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            
            # 对真实图像添加噪声（数据增强）
            real_imgs = real_imgs + torch.randn_like(real_imgs) * noise
            real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

            # 从缓冲区采样并生成负样本
            fake_imgs = buffer.sample_new_exmps(model, steps, step_size, noise, 
                                              batch_size=real_imgs.size(0))

            # 计算正样本和负样本的能量
            real_energy = model(real_imgs)  # 正样本能量（应该低）
            fake_energy = model(fake_imgs)   # 负样本能量（应该高）

            # 计算对比散度损失
            cdiv_loss = real_energy.mean() - fake_energy.mean() 
            
            # 正则化损失（防止能量值过大）
            reg_loss = alpha * (real_energy.pow(2).mean() + fake_energy.pow(2).mean())
            
            # 总损失
            loss = cdiv_loss + reg_loss 

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            total_cdiv_loss += cdiv_loss.item()
            total_reg_loss += reg_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CDiv': f'{cdiv_loss.item():.4f}',
                'Reg': f'{reg_loss.item():.4f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(data_loader)
        avg_cdiv = total_cdiv_loss / len(data_loader)
        avg_reg = total_reg_loss / len(data_loader)
        
        print(f'EBM Epoch {epoch+1}: Loss: {avg_loss:.4f}, '
              f'CDiv: {avg_cdiv:.4f}, Reg: {avg_reg:.4f}')
    
    return model