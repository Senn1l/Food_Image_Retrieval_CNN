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

# Define k values to evaluate
k_values = [3, 5, 11, 21]

# Initialize precision dictionary to store results for each k
precision_results = {k: [] for k in k_values}

total_images = len(imgNames)

# Loop over all images in the dataset
for idx, img_path in enumerate(imgNames):
    if idx % 100 == 0:
      print(f"{idx}/{total_images}")

    # Decode img_path from bytes to string
    img_name = img_path.decode('utf-8')

    # Extract the feature vector for the current image
    queryVec = model.extract_feat(os.path.join(args["database"], img_name), verbose=0)

    # Calculate Euclidean distances
    dists = np.linalg.norm(feats - queryVec, axis=1)
    rank_ID = np.argsort(dists)

    # Get the true class of the query image
    true_class = get_class_name(img_name)

    # Evaluate for each k
    for k in k_values:
        # Get the classes of the top k retrieved images
        retrieved_classes = [get_class_name(imgNames[index].decode('utf-8')) for index in rank_ID[:k]]

        # Count how many of the top k results are correct
        correct_count = retrieved_classes.count(true_class)

        # Calculate precision: correct_count / k
        precision = correct_count / k
        precision_results[k].append(precision)

# Calculate the average precision for each k
for k in k_values:
    average_precision = np.mean(precision_results[k])
    print(f"Average Precision for k={k}: {average_precision:.4f}")