#!/bin/bash

# Convert RAW images to ACES format
# Usage: ./convert_to_aces.sh

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set directories (absolute paths) — see dataset/README.md for the layout
RAW_DIR="$SCRIPT_DIR/../dataset/raw"
ACES_DIR="$SCRIPT_DIR/../dataset/exr/ACES2065-1"
MAX_IMAGES=10

# Create output directory if it doesn't exist
mkdir -p "$ACES_DIR"

# Counter for images processed
count=0

# Process images (recursive: raw/ holds one subdirectory per source dataset)
for input_file in $(find "$RAW_DIR" -type f \( -iname "*.CR2" -o -iname "*.NEF" -o -iname "*.ARW" -o -iname "*.DNG" -o -iname "*.RAF" \) | head -n $MAX_IMAGES); do
  # Convert using rawtoaces 
  rawtoaces --data-dir /usr/local/share/rawtoaces/data --output-dir "$ACES_DIR" --create-dirs --overwrite "$input_file"
  
 
  ((count++))
done

echo "Conversion complete! Processed $count images."
