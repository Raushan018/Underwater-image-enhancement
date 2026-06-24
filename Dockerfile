# Paper §5.3 — Docker deployment
# Single Gunicorn worker to avoid duplicating YOLOv8s-World weights in memory.

FROM python:3.9-slim

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Single worker — avoids RAM duplication of YOLO weights
# 120 s timeout accommodates large-image YOLO inference latency
CMD ["gunicorn", \
     "--workers", "1", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:5000", \
     "app:app"]
