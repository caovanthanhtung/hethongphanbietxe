FROM python:3.10-slim

WORKDIR /app

# system deps for easyocr / opencv (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
