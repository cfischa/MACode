#!/bin/bash

# Define a variable for the current directory path, replacing spaces with underscores or similar to handle special characters.
REPO_PATH=$(pwd | sed 's/ /_/g')

# Create a symbolic link to avoid uppercase or problematic characters
# This will create a symlink within a temporary location
TEMP_DIR="/tmp/docker_symlink"
mkdir -p "$TEMP_DIR"
ln -s "$(pwd)/.." "$TEMP_DIR/repo"

# Run Docker using the symlink to the actual path, without GPU flags
docker run -it --rm \
    -v "$TEMP_DIR/repo:/home/$USER" \
    transfusion \
    /bin/bash -c "cd /home/$USER; exec /bin/bash"

# Clean up the symlink after Docker run is complete
rm -rf "$TEMP_DIR"
