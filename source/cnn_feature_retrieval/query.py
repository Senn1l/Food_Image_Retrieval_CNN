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
ap.add_argument("-query", required=True, help="Path to query which contains image to be queried")
ap.add_argument("-index", required=True, help="Path to index")
ap.add_argument("-model", required=True, choices=["VGG16", "ResNet"], help="Model to use for feature extraction")
ap.add_argument("-database", required=True, help="Path to database")
ap.add_argument("-n", type=int, default=5, help="Number of top images to return")
args = vars(ap.parse_args())

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"], 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# read and show query image
queryDir = args["query"]

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

# extract query image's feature
queryVec = model.extract_feat(queryDir, verbose=1)

# compute Euclidean distance and sort
dists = np.linalg.norm(feats - queryVec, axis=1)
rank_ID = np.argsort(dists)
rank_score = dists[rank_ID]

# number of top retrieved images to show
n = args["n"]
n = max(min(n, 10), 3) # Limit to [3, 10]
imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:n])]
print("top %d images in order are: " % n, imlist)

# loại bỏ dấu nháy
imlist = [img.decode('utf-8').strip("'") for img in imlist]

imgnameList = [get_class_name(filename) for filename in imlist]

# Thêm đường dẫn thư mục database vào từng ảnh
imlist = [os.path.join(args["database"], img) for img in imlist]

# Thêm đường dẫn ảnh query vào đầu imlist
imlist.insert(0, queryDir)

# Đọc các ảnh và lưu vào list images
images = [Image.open(img_path) for img_path in imlist]

# Số ảnh bạn muốn hiển thị (ngoài ảnh query)
n = len(images) - 1

# Sử dụng plt.subplot để tạo result, tạo 3 hàng
fig, axes = plt.subplots(3, (n + 1) // 2, figsize=(7, 5))

# Hiển thị ảnh đầu tiên (ảnh query) ở hàng đầu tiên
axes[0, 2].imshow(images[0])
axes[0, 2].axis('off')
axes[0, 2].text(0.5, -0.1, 'Query Image', ha='center', va='top', transform=axes[0, 2].transAxes, fontsize=12)

# Ẩn các trục còn lại của hàng đầu tiên
for ax in axes[0, 0:]:
    ax.axis('off')

# Hiển thị các ảnh còn lại ở 2 hàng sau
for i, (ax, img) in enumerate(zip(axes[1:].flatten(), images[1:]), 1):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.5, -0.1, f'R{i} ' + imgnameList[i - 1], ha='center', va='top', transform=ax.transAxes, fontsize=12)

queryWithoutExtension = os.path.splitext(queryDir)[0] + "_" + name + "_Result.png"
plt.tight_layout()
plt.savefig(queryWithoutExtension)
plt.close(fig)