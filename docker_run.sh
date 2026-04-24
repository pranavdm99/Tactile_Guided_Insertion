#!/usr/bin/env bash
set -e

XAUTH=/tmp/.docker.xauth
# Create file if it doesn't exist
touch $XAUTH
chmod 644 $XAUTH

# Extract current display's auth cookie and write it
# We use a temp file and cat to avoid breaking Docker's bind-mount inode
temp_auth=$(mktemp)
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$temp_auth" nmerge -

# Explicitly add entries for potential display mismatches (e.g., :0 vs :1)
COOKIE=$(xauth list "$DISPLAY" | head -n 1 | awk '{print $3}')
if [ ! -z "$COOKIE" ]; then
    xauth -f "$temp_auth" add "$(hostname)/unix:0" MIT-MAGIC-COOKIE-1 "$COOKIE"
    xauth -f "$temp_auth" add "$(hostname)/unix:1" MIT-MAGIC-COOKIE-1 "$COOKIE"
fi

cat "$temp_auth" > "$XAUTH"
rm "$temp_auth"

# Run docker compose with correct env
export DISPLAY
export XAUTHORITY=$XAUTH

# Fallback: Allow root to connect if cookies still fail (more specific than xhost +)
# xhost +si:localuser:root > /dev/null 2>&1

docker compose up --build -d --remove-orphans