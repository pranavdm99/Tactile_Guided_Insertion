FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV QT_X11_NO_MITSHM=1

RUN chmod 1777 /tmp && \
    apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libosmesa6-dev \
    libglew-dev \
    libglib2.0-0 \
    libglfw3 \
    patchelf \
    xvfb \
    cmake \
    libegl1 \
    libegl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install CPU-only PyTorch first so robomimic's setup.py sees them already satisfied
# (CPU wheels are ~200 MB vs ~2.5 GB for CUDA wheels)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install custom robosuite fork (robomimic depends on it)
COPY robosuite/ /app/robosuite/
RUN pip install --no-cache-dir /app/robosuite/

# Install custom robomimic fork
COPY robomimic/ /app/robomimic/
RUN pip install --no-cache-dir /app/robomimic/

# Install remaining inference dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir "numpy<2.0"

# Runtime env vars placed after pip layers to preserve build cache
ENV MUJOCO_GL=osmesa
ENV TACTILE_ROOT=/app
ENV PYTHONPATH="/app:/app/env_setup:/app/fots_sim:/app/fots_sim/utils:/app/scripts"

# Copy project code
COPY env_setup/ /app/env_setup/
COPY fots_sim/ /app/fots_sim/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

# Copy trained model checkpoint and dataset
COPY checkpoints/ /app/checkpoints/
COPY datasets/ /app/datasets/

# Ensure output directory exists
RUN mkdir -p /app/output

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default: run 10 inference rollouts with the epoch-2800 model
CMD ["python3", "/app/robomimic/robomimic/scripts/rollout_fots.py", \
     "--agent", "/app/checkpoints/model_epoch_2800.pth", \
     "--n_rollouts", "10", \
     "--horizon", "400", \
     "--video_path", "/app/output/rollout.mp4"]
