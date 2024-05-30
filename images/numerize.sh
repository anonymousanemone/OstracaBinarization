#!/bin/bash

# Counter to keep track of the number
counter=1

# Loop through each file in the current directory
for file in *.JPG; do
    # Check if the item is a file
    if [ -f "$file" ]; then
        # Get the file extension
        extension="${file##*.}"
        # Generate the new filename
        new_filename="original-1-$counter.$extension"
        # Rename the file
        mv "$file" "$new_filename"
        # Increment the counter
        ((counter++))
    fi
done

