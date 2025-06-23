from extract_cnn_vgg16_keras import VGGNet
from extract_cnn_resnet_keras import ResNet

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from PIL import Image
import os

# Function to get the class name from the filename (e.g., apple_pie_0123 -> apple_pie)
def get_class_name(file_name):
    file_without_extension = file_name.split('.')[0]
    parts = file_without_extension.split('_')
    return '_'.join(part for part in parts if not part.isdigit())

ap = argparse.ArgumentParser()
ap.add_argument("-index", required=True, help="Path to index")
ap.add_argument("-model", required=True, choices=["VGG16", "ResNet"], help="Model to use for feature extraction")
ap.add_argument("-database", required=True, help="Path to database")
args = vars(ap.parse_args())

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"], 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

# init VGGNet or ResNet model
name = "VGG16"
if args["model"] == "VGG16":
    model = VGGNet(verbose=1)
elif args["model"] == "ResNet":
    model = ResNet(verbose=1)
    name = "Resnet"
else:
    print("Model not detected, using VGG16")
    model = VGGNet(verbose=1)

def calculate_map_optimized(feats, imgNames, model, database_path, k_values):
    def get_class_name(file_name):
        file_without_extension = file_name.split('.')[0]
        parts = file_without_extension.split('_')
        return '_'.join(part for part in parts if not part.isdigit())

    # Initialize MAP for each k value
    map_results = {k: 0 for k in k_values}
    max_k = max(k_values)  # Get the maximum k value

    total_images = len(imgNames)

    # Loop over all images in the dataset
    for idx, img_path in enumerate(imgNames):
        if idx % 100 == 0:
            print(f"{idx}/{total_images}")
        
        # Decode img_path from bytes to string
        img_name = img_path.decode('utf-8')
        
        # Extract the feature vector for the current image
        queryVec = model.extract_feat(os.path.join(database_path, img_name), verbose=0)
        
        # Calculate Euclidean distances and sort them
        dists = np.linalg.norm(feats - queryVec, axis=1)
        rank_ID = np.argsort(dists)[:max_k]  # Get top max_k results

        # Get the true class of the query image
        true_class = get_class_name(img_name)
        
        # Track relevant count and precision at each k
        relevant_count = 0
        precision_at_k = []

        # Compute precision at each rank up to max_k
        for i in range(max_k):
            retrieved_class = get_class_name(imgNames[rank_ID[i]].decode('utf-8'))
            
            if retrieved_class == true_class:
                relevant_count += 1
                precision_at_k.append(relevant_count / (i + 1))
        
        # Calculate AP for each k in k_values
        for k in k_values:
            if relevant_count > 0 and len(precision_at_k) >= k:
                average_precision = np.mean(precision_at_k[:k])
            else:
                average_precision = 0
            
            map_results[k] += average_precision

    # Calculate final MAP for each k
    num_queries = len(imgNames)
    for k in k_values:
        map_results[k] /= num_queries
        print(f"MAP@{k}: {map_results[k]:.4f}")

# Usage
k_values = [3, 5, 11, 21]
calculate_map_optimized(feats, imgNames, model, args["database"], k_values)