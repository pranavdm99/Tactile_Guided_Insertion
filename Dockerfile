FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV MUJOCO_GL=egl
ENV QT_X11_NO_MITSHM=1

# TACTILE_ROOT tells rollout_fots.py where env_setup/ and fots_sim/ live
ENV TACTILE_ROOT=/app

ENV PYTHONPATH="/app:/app/env_setup:/app/fots_sim:/app/fots_sim/utils:/app/scripts"

RUN chmod 1777 /tmp && \
    apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libosmesa6-dev \
    libglew-dev \
    libglib2.0-0 \
    patchelf \
    xvfb \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install custom robosuite fork (robomimic depends on it)
COPY robosuite/ /app/robosuite/
RUN pip install --no-cache-dir -e /app/robosuite/

# Install custom robomimic fork
COPY robomimic/ /app/robomimic/
RUN pip install --no-cache-dir -e /app/robomimic/

# Install remaining inference dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir "numpy<2.0"

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
