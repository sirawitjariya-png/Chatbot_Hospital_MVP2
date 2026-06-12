FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py server.py ./
COPY app/ ./app/
COPY data/ ./data/

# Cloud Run injects PORT (default 8080); main.py reads it
ENV PORT=8080

CMD ["python", "main.py", "serve"]
