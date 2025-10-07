# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*


# Allow statements and log messages to be sent straight to the terminal.
ENV PYTHONUNBUFFERED=TRUE

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the working directory.
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory.
COPY . .

# Gunicorn is used to run the application on a production server.
# It listens on port 8080.
CMD ["python", "main.py"]