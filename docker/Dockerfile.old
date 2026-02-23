FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "numpy<2" \
    opencv-python-headless \
    tflite-runtime \
    fastapi \
    "uvicorn[standard]"

# server
COPY scripts/server.py /app/server.py

EXPOSE 8000

# Környezeti változókkal állítjuk a base/model/min_conf értékeket
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

