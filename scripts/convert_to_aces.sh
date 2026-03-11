#!/bin/bash

# Convert RAW images to ACES format
# Usage: ./convert_to_aces.sh

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set directories (absolute paths)
RAW_DIR="$SCRIPT_DIR/../dataset/temp/raw"
ACES_DIR="$SCRIPT_DIR/../dataset/temp/aces"
MAX_IMAGES=10

# Create output directory if it doesn't exist
mkdir -p "$ACES_DIR"

# Counter for images processed
count=0

# Process images
for image in $(ls "$RAW_DIR" | head -n $MAX_IMAGES); do
  input_file="$RAW_DIR/$image"
  
  # Skip if not a file
  [ ! -f "$input_file" ] && continue
  
  # Convert using rawtoaces 
  rawtoaces --data-dir /usr/local/share/rawtoaces/data --output-dir "$ACES_DIR" --create-dirs --overwrite "$input_file"
  
 
  ((count++))
done

echo "Conversion complete! Processed $count images."
