FROM python:3.11-slim

# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies for scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
# Added scipy and numpy explicitly as gui_smooth likely uses more complex fitting/filtering
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    matplotlib \
    openpyxl \
    scipy \
    numpy

# Copy application code
COPY . .

EXPOSE 8501

# Updated to use gui_smooth.py
ENTRYPOINT ["streamlit", "run", "gui_smooth.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
