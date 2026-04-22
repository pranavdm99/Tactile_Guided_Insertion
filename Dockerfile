FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    tmux \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    libglib2.0-0 \
    patchelf \
    xvfb \
    freeglut3-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

RUN pip install "numpy<2.0"

# EGL / X11 rendering variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV MUJOCO_GL=egl
ENV QT_X11_NO_MITSHM=1
ENV PYTHONPATH="/app:/app/env_setup:/app/fots_sim:/app/fots_sim/utils"

# Set PYTHONPATH
RUN echo 'export PYTHONPATH="/app:/app/env_setup:/app/fots_sim:/app/fots_sim/utils"' >> /root/.bashrc

# Configure Entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
