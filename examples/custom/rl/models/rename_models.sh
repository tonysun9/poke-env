#!/bin/bash

# Loop through all files in the current directory
for filename in *; do
  # Check if the filename contains "observation"
  if [[ "$filename" == *"observation"* ]]; then
    # Extract the filename without the extension
    base_filename="${filename%.*}"

    # Replace "observation" with "obs"
    new_filename="${base_filename/observation/obs}.${filename##*.}"

    # Move the file to the new filename
    mv "$filename" "$new_filename"

    # Optional: Print a message
    echo "Renamed file: $filename -> $new_filename"
  fi
done