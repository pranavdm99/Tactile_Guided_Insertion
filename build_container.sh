#!/bin/bash
# Build script for Tactile Insertion Apptainer image

IMAGE_NAME="tactile_insertion.sif"
DOCKERFILE="Dockerfile"

echo "Checking for existing image..."
if [ -f "$IMAGE_NAME" ]; then
    echo "Image $IMAGE_NAME already exists. Skipping build."
    exit 0
fi

echo "Starting Apptainer build from $DOCKERFILE..."
module load apptainer

# We use --fakeroot if available, otherwise standard build
# We also set a cache directory in scratch to avoid filling up home quota
export APPTAINER_CACHEDIR=$HOME/scratch/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR

apptainer build --fakeroot $IMAGE_NAME $DOCKERFILE

if [ $? -eq 0 ]; then
    echo "✅ Build successful: $IMAGE_NAME"
else
    echo "❌ Build failed. Check the logs above."
    exit 1
fi
