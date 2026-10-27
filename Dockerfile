FROM python:3.12-slim

# Install Tesseract OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static/ static/

# Persistent data directory
ENV BOOKSCANNER_DATA=/app/data
RUN mkdir -p /app/data

EXPOSE 5001

CMD ["python3", "app.py"]
