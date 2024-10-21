import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import sobel

import os
import glob

def get_crop_bundaries(data):

    # We assume that the brain occupies higher intensity values in the image
    # Summing up pixel values along the z-axis (depth) to find the slice with the most brain tissue
    sums = np.sum(np.sum(data, axis=0), axis=0)

    # Find the index of the slice with the maximum sum
    max_index = np.argmax(sums)

    # Select the slice with the most brain tissue
    slice_data = data[:, :, max_index]

    percentage_row = 0.15
    percentage_col = 0.17

    # Initialize limits
    top_limit = 0
    bottom_limit = slice_data.shape[0] - 1
    left_limit = 0
    right_limit = slice_data.shape[1] - 1



    means_row = np.array([np.mean(slice_data[row_idx, :]) for row_idx in range(slice_data.shape[0])])
    threshold_row = percentage_row * np.max(means_row)

    # Determine the top limit
    for row_idx in range(slice_data.shape[0]):
        row_mean = np.mean(slice_data[row_idx, :])
        if not row_mean < threshold_row:
            top_limit = row_idx
            break

    # Determine the bottom limit
    for row_idx in range(slice_data.shape[0] - 1, -1, -1):
        row_mean = np.mean(slice_data[row_idx, :])
        if not row_mean < threshold_row:
            bottom_limit = row_idx
            break



    means_col = np.array([np.mean(slice_data[:, col_idx]) for col_idx in range(slice_data.shape[1])])
    threshold_col = percentage_col * np.max(means_col)

    # Determine the left limit
    for col_idx in range(slice_data.shape[1]):
        col_mean = np.mean(slice_data[:, col_idx])
        if not col_mean < threshold_col:
            left_limit = col_idx
            break

    # Determine the right limit
    for col_idx in range(slice_data.shape[1] - 1, -1, -1):
        col_mean = np.mean(slice_data[:, col_idx])
        if not col_mean < threshold_col:
            right_limit = col_idx
            break

    return top_limit, bottom_limit, left_limit, right_limit


def get_front_back_limits(array, threshold, num_slices):

	front_limit = 0
	for i in range(num_slices):
		if array[i] > threshold:
			front_limit = i
			break


	back_limit = 0
	for i in range(num_slices-1, -1, -1):
		if array[i] > threshold:
			back_limit = i
			break


	return front_limit, back_limit
    

def sobel_edges_front_back(percentage, data):

    plt.figure(figsize=(12, 8))

    num_slices = data.shape[2]

    # Calculate the sum of edge responses for each slice using Sobel
    edges = [np.sum(sobel(data[:, :, i])) for i in range(num_slices)]

    threshold = percentage * np.max(edges)

    front_limit, back_limit = get_front_back_limits(edges, threshold, num_slices)
        
    return front_limit, back_limit



def preprocess_image(file_path, destination=None, labels_dest=None, dataframe=None, labels=True):

    # Load the MRI image    
    img = nib.load(file_path)
    data = img.get_fdata()

    top_limit, bottom_limit, left_limit, right_limit = get_crop_bundaries(data)
    front_limit, back_limit = sobel_edges_front_back(0.65, data)

    # Crop all layers of the 3D image using the calculated boundaries
    cropped_data = data[top_limit:bottom_limit+1, left_limit:right_limit+1, front_limit:back_limit+1]

    if not labels:
        return cropped_data

    # Create a new NIfTI image
    cropped_img = nib.Nifti1Image(cropped_data, img.affine, img.header)

    # Save the new cropped image
    image_id = file_path.split('/')[-1]
    output_file_path = os.path.join(destination, image_id)
    nib.save(cropped_img, output_file_path)

    for ind in dataframe.index:
        label = 0 #AD
        if (dataframe["Group"][ind] == "MCI"):
            label = 1
        elif (dataframe["Group"][ind] == "CN"):
            label = 2

        if os.path.exists(output_file_path):
            labels_dest.write(f"{output_file_path},{label}\n")



def preprocess_folder(source_folder, output_folder, labels_source, labels_destination):

    f = open(os.path.join(labels_destination))

    files = [f for f in os.listdir(source_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
    i=0

    df = pd.read_csv(labels_source)

    for file in files:
        file_path = os.path.join(source_folder, file)
        preprocess_image(file_path, output_folder, f, df)
        i += 1

        exists = glob.glob(f'{output_folder}/{file}')

        if len(exists) == 0:
            print(f'File {file} not found')
            break
        else:
            print(f'{file}') 