import os
import h5py
import numpy as np
import argparse

from extract_cnn_vgg16_keras import VGGNet
from extract_cnn_resnet_keras import ResNet

ap = argparse.ArgumentParser()
ap.add_argument("-database", required=True, help="Path to database which contains images to be indexed")
ap.add_argument("-index", required=True, help="Name of index file")
ap.add_argument("-model", required=True, choices=["VGG16", "ResNet"], help="Model to use for feature extraction")
args = vars(ap.parse_args())

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

if __name__ == "__main__":
    db = args["database"]
    img_list = get_imlist(db)

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    if args["model"] == "VGG16":
        model = VGGNet()
    elif args["model"] == "ResNet":
        model = ResNet()
    else:
        model = VGGNet() # not found

    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))

    feats = np.array(feats)
    output = args["index"]

    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()