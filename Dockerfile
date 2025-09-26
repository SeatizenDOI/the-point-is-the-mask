# Use NVIDIA's CUDA base image with Ubuntu and Python 3.10
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Set environment variables 
# not to prompt for user input during package installation.
ENV DEBIAN_FRONTEND=noninteractive 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-dev \
    cmake \
    build-essential \
    libeigen3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    geopandas==1.0.1 \
    huggingface-hub==0.31.2 \
    numpy==1.26.0 \
    opencv-python==4.11.0.86 \
    pandas==2.2.3 \
    pillow==11.2.1 \
    pyproj==3.7.1 \
    pytz==2025.2 \
    rasterio==1.4.3 \
    requests==2.32.3 \
    scikit-learn==1.6.1 \
    scipy==1.15.3 \
    segment-anything-py==1.0.1 \
    shapely==2.1.0 \
    tifffile==2025.5.10 \
    torch==2.3.1 \
    transformers==4.51.3 \
    fastgeodis==1.0.5 \
    tqdm==4.67.1 \
    matplotlib==3.10.1 \
    datasets==3.5.0 \
    seaborn==0.13.2 \
    accelerate==1.6.0 \
    tensorboardX==2.6.2.2

    
RUN rm /usr/local/lib/python3.12/dist-packages/FastGeodisCpp.cpython-312-x86_64-linux-gnu.so
COPY ./dependencies/FastGeodisCpp.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/

# Add local directory and change permission.
ADD --chown=seatizen . /home/seatizen/app/

# Setup workdir in directory.
WORKDIR /home/seatizen/app


# Change with our user.
USER seatizen

# Define the entrypoint script to be executed.
CMD ["/bin/bash"]