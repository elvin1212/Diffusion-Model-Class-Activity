import torch.nn as nn
import torch

class SimpleDiffusionModel(nn.Module):
    def __init__(self, image_size=32, num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        
        # CNN architecture for noise prediction
        self.network = nn.Sequential(
            nn.Conv2d(num_channels + 1, 64, 3, padding=1),  # +1 for time embedding
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_channels, 3, padding=1)  # Output noise
        )
        
    def forward(self, x, t):
        """
        x: input image tensor [batch, channels, height, width]
        t: time step tensor [batch] or scalar
        """
        # Ensure t has correct shape [batch, 1, 1, 1] for broadcasting
        if isinstance(t, (int, float)):
            t = torch.tensor([t] * x.size(0), device=x.device)
        
        t = t.view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
        t_expanded = t.expand(-1, 1, x.size(2), x.size(3))  # [batch, 1, height, width]
        
        # Concatenate time embedding with input image
        x_with_time = torch.cat([x, t_expanded], dim=1)  # [batch, channels+1, height, width]
        
        return self.network(x_with_time)

def get_model(model_name, **kwargs):
    if model_name == "Diffusion":
        image_size = kwargs.get('image_size', 32)  # CIFAR-10 image size is 32x32
        num_channels = kwargs.get('num_channels', 3)
        return SimpleDiffusionModel(image_size=image_size, num_channels=num_channels)
    else:
        raise ValueError(f"Model {model_name} not supported")