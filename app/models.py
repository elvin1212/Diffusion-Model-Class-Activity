import torch
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from diffusion.model import get_model as get_diffusion_model
from EBM.model import get_model as get_ebm_model

class ModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_model = None
        self.ebm_model = None
        
    def load_diffusion_model(self, model_path="diffusion/diffusion_model.pth"):
        """
        Load trained diffusion model
        """
        try:
            # Create model instance
            self.diffusion_model = get_diffusion_model("Diffusion", image_size=32, num_channels=3)
            self.diffusion_model.to(self.device)
            
            # Load model weights
            model_path = os.path.join(os.path.dirname(__file__), '..', model_path)
            if os.path.exists(model_path):
                self.diffusion_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.diffusion_model.eval()
                print(f"Successfully loaded Diffusion model: {model_path}")
            else:
                print(f"Warning: Diffusion model file does not exist: {model_path}")
                
        except Exception as e:
            print(f"Error loading Diffusion model: {e}")
            self.diffusion_model = None
            
        return self.diffusion_model
    
    def load_ebm_model(self, model_path="EBM/ebm_model.pth"):
        """
        Load trained EBM model
        """
        try:
            # Create model instance
            self.ebm_model = get_ebm_model("EBM", image_size=32, num_channels=3)
            self.ebm_model.to(self.device)
            
            # Load model weights
            model_path = os.path.join(os.path.dirname(__file__), '..', model_path)
            if os.path.exists(model_path):
                self.ebm_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.ebm_model.eval()
                print(f"Successfully loaded EBM model: {model_path}")
            else:
                print(f"Warning: EBM model file does not exist: {model_path}")
                
        except Exception as e:
            print(f"Error loading EBM model: {e}")
            self.ebm_model = None
            
        return self.ebm_model
    
    def get_diffusion_model(self):
        """
        Get loaded diffusion model
        """
        return self.diffusion_model
    
    def get_ebm_model(self):
        """
        Get loaded EBM model
        """
        return self.ebm_model
    
    def get_device(self):
        """
        Get current device
        """
        return self.device