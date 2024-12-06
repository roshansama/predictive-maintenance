FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements file into the container
COPY requirements_2.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_2.txt

# Copy all files into the container
COPY . .

# Specify the default command to run the application
CMD ["python", "app.py"]
