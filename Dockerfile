# 1. Start with a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install dependencies
# We do this first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/app:$PYTHONPATH

# 6. Expose the port FastAPI will run on
EXPOSE 8000

# 7. Default Command: Start the API
# src.main:app refers to the 'app' object inside src/main.py
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]