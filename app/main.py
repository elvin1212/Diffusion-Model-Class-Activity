from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import io
import base64
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import random
import glob

from app.models import ModelLoader

app = FastAPI(title="CIFAR-10 Diffusion and EBM Models API")

# Initialize model loader
model_loader = ModelLoader()

# Load models
model_loader.load_diffusion_model()
model_loader.load_ebm_model()

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class GenerationRequest(BaseModel):
    num_samples: int = 6
    diffusion_steps: int = 100  # For Diffusion model only
    langevin_steps: int = 1000  # For EBM model only
    step_size: float = 0.01    # For EBM model only
    noise: float = 0.005       # For EBM model only

class GenerationResponse(BaseModel):
    images: List[str]  # Base64 encoded images
    model_used: str

@app.get("/")
async def read_root():
    return {
        "message": "CIFAR-10 Diffusion and EBM Models API",
        "endpoints": {
            "/": "Welcome endpoint",
            "/generate/diffusion": "Generate images using Diffusion model",
            "/generate/ebm": "Generate images using EBM model",
            "/show": "Show generated images page",
            "/health": "Health check"
        }
    }

@app.get("/show")
async def show_images():
    return FileResponse("show_images.html")

@app.post("/generate/diffusion", response_model=GenerationResponse)
@app.get("/generate/diffusion", response_model=GenerationResponse)
async def generate_with_diffusion(request: GenerationRequest = None):
    """
    Generate images using Diffusion model
    """
    # Use default parameters for GET requests
    if request is None:
        request = GenerationRequest()
        
    try:
        model = model_loader.get_diffusion_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Diffusion model not loaded")
            
        device = model_loader.get_device()
        
        # Generate samples
        samples = await _generate_diffusion_samples(
            model, 
            device, 
            request.num_samples, 
            request.diffusion_steps
        )
        
        # Convert to Base64 images
        images_b64 = _convert_images_to_base64(samples)
        
        return GenerationResponse(images=images_b64, model_used="diffusion")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/ebm", response_model=GenerationResponse)
@app.get("/generate/ebm", response_model=GenerationResponse)
async def generate_with_ebm(request: GenerationRequest = None):
    """
    Generate images using EBM model
    """
    # Use default parameters for GET requests
    if request is None:
        request = GenerationRequest()
        
    try:
        model = model_loader.get_ebm_model()
        if model is None:
            raise HTTPException(status_code=500, detail="EBM model not loaded")
            
        device = model_loader.get_device()
        
        # Generate samples
        samples = await _generate_ebm_samples(
            model, 
            device, 
            request.num_samples, 
            request.langevin_steps,
            request.step_size,
            request.noise
        )
        
        # Convert to Base64 images
        images_b64 = _convert_images_to_base64(samples)
        
        return GenerationResponse(images=images_b64, model_used="ebm")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

async def _generate_diffusion_samples(model, device, num_samples, diffusion_steps):
    """
    Generate Diffusion samples
    """
    model.eval()
    
    # Create random noise - initial samples (no gradient required)
    samples = torch.randn(num_samples, 3, 32, 32, device=device)
    
    # Reverse diffusion process
    for step in range(diffusion_steps):
        # Current time step (from 1 to 0)
        current_time = 1.0 - step / diffusion_steps
        t = torch.full((num_samples,), current_time, device=device)
        
        # Key fix: Create a copy that requires gradient, instead of modifying the original tensor
        samples_with_grad = samples.clone().requires_grad_(True)
        
        # Predict noise
        predicted_noise = model(samples_with_grad, t)
        
        # Manually calculate gradient: objective is to minimize predicted noise
        noise_loss = torch.mean(predicted_noise ** 2)
        
        # Calculate gradient (only for samples_with_grad)
        gradients = torch.autograd.grad(
            outputs=noise_loss,
            inputs=samples_with_grad,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Apply gradient descent step to original samples
        learning_rate = 0.1 / (diffusion_steps * (1 - current_time + 1e-5))  # Adaptive learning rate
        samples = samples - learning_rate * gradients.detach()
        
        # Add small noise (except for the last step)
        if step < diffusion_steps - 1:
            noise_scale = 0.5 * (1 - current_time)  # Increased noise at early steps
            samples = samples + noise_scale * torch.randn_like(samples)
        
        # Numerical stability: clip values
        samples = torch.clamp(samples, -1.0, 1.0)
    
    # Convert to image format [0, 1] range
    samples = (samples + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    
    return samples.detach()

async def _generate_ebm_samples(model, device, num_samples, steps, step_size, noise):
    """
    Generate samples using EBM model
    """
    model.eval()
    
    # Start from random noise with proper normalization
    samples = torch.randn(num_samples, 3, 32, 32, device=device)
    samples = samples / (samples.norm(dim=1, keepdim=True) + 1e-8)  # Normalize
    
    # Langevin Sampling process with adaptive step size
    for step in range(steps):
        samples.requires_grad_(True)
        energy = model(samples)
        grad = torch.autograd.grad(energy.sum(), samples, create_graph=False)[0]
        
        # Update samples with adaptive step size
        current_step_size = step_size * (1.0 - step / steps)  # Decreasing step size
        samples = samples - current_step_size * grad + noise * torch.randn_like(samples)
        
        # Clip value range
        samples = torch.clamp(samples, -1.0, 1.0)
        
        # No gradient needed for next calculation
        samples = samples.detach()
        
        # Occasionally re-normalize to prevent divergence
        if step % 100 == 0 and step > 0:
            samples = samples / (samples.norm(dim=1, keepdim=True) + 1e-8)
    
    # Convert to image format [0, 1] range (denormalization)
    samples = (samples + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    
    return samples

def _convert_images_to_base64(samples):
    """
    Convert PyTorch tensors to Base64 encoded PNG images
    """
    images_b64 = []
    
    for i in range(samples.shape[0]):
        # Get single image
        img_tensor = samples[i]  # [C, H, W]
        
        # Convert to numpy array and adjust dimension order
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        
        # Ensure values are in [0, 1] range and convert to [0, 255]
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Create PIL image
        img_pil = Image.fromarray(img_np)
        
        # Convert to Base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        images_b64.append(img_b64)
    
    return images_b64

# To run locally: uvicorn app.main:app --reload
# To run with Docker: docker build -t cifar10-app . && docker run -p 8000:8000 cifar10-app
# Or with docker-compose: docker-compose up --build
