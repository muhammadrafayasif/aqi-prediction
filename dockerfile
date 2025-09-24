# Start from Python slim image
FROM python:3.11-slim

# Install system dependencies (needed for Playwright + Firefox)
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    git \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxrandr2 \
    libxdamage1 \
    libgbm1 \
    libpango-1.0-0 \
    libgtk-3-0 \
    firefox-esr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       "hopsworks[python]" \
       pandas \
       playwright \
       python-dotenv

# Install Playwright browsers (only Firefox)
RUN playwright install --with-deps firefox

# Set working directory
WORKDIR /app

# Copy code into container (optional if mounting in CI/CD)
COPY . /app

# Default command (override in GitHub Actions)
CMD ["python3"]
