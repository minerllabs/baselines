FROM chainer/chainer:v6.3.0-python3

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Install Minecraft needed libraries
RUN apt-get update
RUN apt-get install openjdk-8-jdk git -y

# Install requirements
COPY ./requirements.txt .
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt

# Copy script
COPY . .

# Install X server
RUN apt-get update && apt-get install -y \
    libx11-6 \
    xvfb \
    python-opengl \
    x11-xserver-utils \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    mesa-utils \
 && rm -rf /var/lib/apt/lists/*
