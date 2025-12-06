FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY ./diffusion/requirements.txt /app/diffusion_requirements.txt
COPY ./EBM/requirements.txt /app/ebm_requirements.txt
RUN pip install --no-cache-dir -r diffusion_requirements.txt
RUN pip install --no-cache-dir -r ebm_requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn pydantic pillow

COPY ./app /app/app
COPY ./diffusion /app/diffusion
COPY ./EBM /app/EBM
COPY ./show_images.html /app/show_images.html
COPY ./test_imgs /app/test_imgs

RUN mkdir -p /app/diffusion /app/EBM

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]