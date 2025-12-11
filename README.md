# CIFAR-10 Diffusion and EBM Models API

This project demonstrates image generation using two different approaches: Diffusion Models and Energy-Based Models (EBM) on the CIFAR-10 dataset.

## Project Structure

- `app/` - FastAPI application
- `diffusion/` - Diffusion model implementation
- `EBM/` - Energy-Based Model implementation
- `test_imgs/` - CIFAR-10 test images organized by class
- `show_images.html` - Web interface to visualize generated images

## Running with Docker

### Using Docker Compose (Recommended)

To build and run the application using Docker Compose:

```bash
docker-compose up --build
```

This command will:
- Build the Docker image from the Dockerfile
- Start the container and expose port 8000
- Make the application available at http://localhost:8000

To stop the containers, use:
```bash
docker-compose down
```

To rebuild the image and restart the containers:
```bash
docker-compose up --build --force-recreate
```

### Using Docker directly

1. Build the Docker image:
```bash
docker build -t cifar10-app .
```

2. Run the container:
```bash
docker run -p 8000:8000 cifar10-app
```

The application will be available at http://localhost:8000

Additional Docker options:
- To run in detached mode: `docker run -d -p 8000:8000 cifar10-app`
- To access container shell: `docker run -it -p 8000:8000 --entrypoint /bin/bash cifar10-app`
- To remove container after exit: `docker run --rm -p 8000:8000 cifar10-app`

## API Endpoints

- `GET /` - Welcome message and API information
- `GET /show` - Web interface to visualize generated images
- `GET/POST /generate/diffusion` - Generate images using Diffusion model
- `GET/POST /generate/ebm` - Generate images using EBM model
- `GET /health` - Health check endpoint

## Running Locally

1. Install dependencies:
```bash
pip install torch torchvision matplotlib numpy tqdm fastapi uvicorn pydantic pillow
```

2. Run the application:
```bash
uvicorn app.main:app --reload
```

The application will be available at http://localhost:8000