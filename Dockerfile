# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements_2.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_2.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the application port
EXPOSE 8501

# Specify the default command to run the application
CMD ["python", "app.py"]
