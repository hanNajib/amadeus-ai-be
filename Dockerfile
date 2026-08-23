FROM python:3.10-slim

# Install dependency audio dan build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

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

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Tambahkan instalasi cloudflared
RUN wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && \
    dpkg -i cloudflared-linux-amd64.deb && \
    rm cloudflared-linux-amd64.deb

# Jalankan uvicorn dan tunnel bersamaan
CMD cloudflared tunnel --no-autoupdate run --token TOKEN_DARI_CLOUDFLARE_KAMU & uvicorn main:app --host 0.0.0.0 --port 8000