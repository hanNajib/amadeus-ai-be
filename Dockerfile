FROM python:3.10-slim

# Install dependency audio dan build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Unduh binary standalone cloudflared
ADD https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 /usr/local/bin/cloudflared
RUN chmod +x /usr/local/bin/cloudflared

WORKDIR /app

# Copy requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model bobot dari Hugging Face ke dalam container
COPY download_weights.py .
RUN python download_weights.py

# Copy sisa project
COPY . .

EXPOSE 8000

# Jalankan Cloudflare Tunnel di background lalu start Uvicorn
CMD /usr/local/bin/cloudflared tunnel --no-autoupdate run --token eyJhIjoiMTVmMjRiOTE2ODY4ZjU4OTZmZDQwMjQ5YmI1NzQ2YzEiLCJ0IjoiNmVhZjhlYTgtZTRiYy00ODQwLTk1YmEtOGFjNWJmOGM5NzVlIiwicyI6Ik1XTTJaVEkwWWpBdE9USTJOeTAwTlRRM0xUZzVZemt0Wm1NNVlUWXdOelV6WXpZMSJ9 & uvicorn main:app --host 0.0.0.0 --port 8000