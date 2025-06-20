# 1. Builder stage: To install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# First, install the CPU-only version of PyTorch to keep the image size down
RUN pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

# Copy and install the rest of the requirements into the virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# 2. Final stage: The actual application image
FROM python:3.11-slim

WORKDIR /code

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the path to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY . .

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 