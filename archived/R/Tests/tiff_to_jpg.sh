#!/bin/bash
for file in *.tiff; do
  mv "$file" "${file%.tiff}.jpg"
done
