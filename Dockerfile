# Step 1: Use an official, lightweight Python base image optimized for data science workloads
FROM python:3.11-slim

# Step 2: Set environmental variables to optimize Python performance inside the container
# PYTHONDONTWRITEBYTECODE=1 prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED=1 prevents Python from buffering stdout/stderr, ensuring live log streaming
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Step 3: Establish the active working directory inside the container's file system
WORKDIR /app

# Step 4: Install system-level dependencies required for basic system operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Copy over our requirements map and leverage pip cache optimization layers
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Step 6: Copy over our local code and pre-trained neural network parameters
COPY main.py /app/
COPY server_lstm_model.keras /app/

# Step 7: Expose the virtual networking port that FastAPI listens to
EXPOSE 8000

# Step 8: Define the runtime execution entry-point to run our production ASGI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]