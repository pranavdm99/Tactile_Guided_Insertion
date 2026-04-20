#!/usr/bin/env bash
set -e

XAUTH=/tmp/.docker.xauth

# Create file if it doesn't exist
touch $XAUTH
chmod 600 $XAUTH

# Extract current display's auth cookie and write it
xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Run docker compose with correct env
export DISPLAY
export XAUTHORITY=$XAUTH

docker compose up --build -d --remove-orphans