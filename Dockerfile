# Deepfake Detector — multi-stage Dockerfile
# Build:   docker build -t deepfake-detector .
# Run:     docker run -p 8501:8501 deepfake-detector

FROM python:3.11-slim AS base

# System deps for OpenCV / facenet-pytorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Streamlit health probe / production server settings
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Entrypoint — bind to 0.0.0.0 so it's reachable outside the container
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
