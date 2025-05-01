# Use an official Python runtime as a parent image
# Using slim-bullseye for a smaller image size. Choose a version compatible with your dependencies.
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Hugging Face cache to keep it inside the container (and potentially mountable)
# This helps avoid re-downloading models on every start if a volume is mounted.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub

# Prevent Python from writing pyc files to disc (optional)
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to terminal (useful for logs)
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be needed (e.g., for some Python packages)
# Example: git might be needed if pip installs from git repos
# RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Make sure pip is up-to-date
RUN pip install --upgrade pip
# Install dependencies, including torch with CUDA support (ensure base image OS/CUDA compatibility if needed)
# The torch install command comes from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the cache directory and set permissions (optional, helps if running non-root)
# RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME

# Make port 8501 available to the world outside this container (Streamlit default)
EXPOSE 8501

# Define the command to run your app using streamlit
# Healthcheck can be added for more robust deployments
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 