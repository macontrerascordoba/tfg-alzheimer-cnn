#!/bin/bash

# Specify the path to the main folder
container_folder="/home/miguel/Documents/tfg-alzheimer-cnn/.data/Image_Collections/ADNI1_Annual_2_Yr_3T/ADNI"
destination_folder="/home/miguel/Documents/tfg-alzheimer-cnn/.data/Image_Collections/ADNI1_Annual_2_Yr_3T"

# Find all .nii files and move them to the main folder
find "$container_folder" -type f -name "*.nii" -exec mv {} "$destination_folder" \;

# Rename the .nii files in the destination folder
for file in "$destination_folder"/*.nii; do
    if [ -f "$file" ]; then
        # Get the part after the last "_" in the filename
        new_name=$(echo "$file" | awk -F_ '{print $NF}')

        # Rename the file
        mv "$file" "$destination_folder/$new_name"
    fi
done


# Delete all directories, excluding the main folder
find "$destination_folder" -mindepth 1 -type d -exec rm -r {} \;s

echo "All .nii files have been moved and renamed in $destination_folder and all directories have been deleted."
