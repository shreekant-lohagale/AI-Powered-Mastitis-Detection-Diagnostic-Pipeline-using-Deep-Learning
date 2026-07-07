# 1. Use an official, lightweight Python runtime as a parent image
FROM python:3.12-slim

# 2. Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install system-level dependencies required by OpenCV/TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy the requirements file into the container
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the entire project context into the container
COPY . .

# 8. Define the default command to execute the evaluation script
CMD ["python", "src/evaluate.py"]