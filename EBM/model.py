import torch
import torch.nn as nn

class EBM(nn.Module):
    def __init__(self, image_size=32, num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        
        # Simple CNN architecture for energy calculation
        self.network = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """
        x: input image tensor [batch, channels, height, width]
        Returns: energy values [batch, 1]
        """
        return self.network(x)

def get_model(model_name, **kwargs):
    if model_name == "EBM":
        image_size = kwargs.get('image_size', 32)
        num_channels = kwargs.get('num_channels', 3)
        return EBM(image_size=image_size, num_channels=num_channels)
    else:
        raise ValueError(f"Model {model_name} not supported")