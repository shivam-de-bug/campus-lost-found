FROM python:3.10-slim

# Install system dependencies for FAISS and image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy the requirements file first to leverage Docker cache
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY --chown=user . $HOME/app

# Ensure directories exist
RUN mkdir -p found_items lost_items static

# Expose the standard Hugging Face Space port
EXPOSE 7860

# Start the application
CMD ["python", "main.py"]
