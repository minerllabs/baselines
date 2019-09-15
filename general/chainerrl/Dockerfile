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
    python-opengl\
    x11-xserver-utils\
 && rm -rf /var/lib/apt/lists/*

# Install jupyter notebook
RUN pip install jupyter pandas matplotlib numpy scipy sklearn

# Create starting file
RUN echo "xhost + & jupyter notebook --allow-root --ip 0.0.0.0" > /app/xvfb.sh
RUN echo "xvfb-run -s \"-screen 0 1400x900x24\" /app/xvfb.sh" > /app/run.sh
RUN chmod ugo+x /app/xvfb.sh
RUN chmod ugo+x /app/run.sh

# Set the default command to jupyter
CMD ["sh", "/app/run.sh"]
