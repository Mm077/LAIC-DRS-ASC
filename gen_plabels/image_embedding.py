import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import copy
import torch
import data_utils
import numpy as np
from models import CLIPModel

dataset = "UCF101" #　["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
                   #  ["EuroSAT"]

dataloader_train, dataloader_test = data_utils.get_dataloader(
    dataset=dataset, batch_size=1024
)

backbone = 'ViT-B/32' # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
data_dir = 'data' # 'data-b32', 'data-b16', 'data-l14'
model = CLIPModel(model_name=backbone).cuda()
model.eval()

features = []
labels = []
print("Inferring image features and labels...")
for iteration, (x, y) in enumerate(dataloader_train):
    x = x.cuda()
    with torch.no_grad():
        feature = model.encode_image(x)
    features.append(feature.cpu().numpy())
    labels.append(y.numpy())
    if iteration % 10 == 0:
        print(f"[Iter {iteration}/{len(dataloader_train)}]")
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)
print("Feature shape:", features.shape, "Label shape:", labels.shape)

features_test = []
labels_test = []

print("Inferring test image features and labels...")
for iteration, (x, y) in enumerate(dataloader_test):
    x = x.cuda()
    with torch.no_grad():
        feature = model.encode_image(x)
    features_test.append(feature.cpu().numpy())
    labels_test.append(y.numpy())
    if iteration % 10 == 0:
        print(f"[Iter {iteration}/{len(dataloader_test)}]")
features_test = np.concatenate(features_test, axis=0)
labels_test = np.concatenate(labels_test, axis=0)
print("Feature shape:", features_test.shape, "Label shape:", labels_test.shape)


os.makedirs(f"./{data_dir}/" + f"{dataset}/", exist_ok=True)

np.save(f"./{data_dir}/" + f"{dataset}/" + dataset  + "_image_embedding_train.npy", features)
np.save(f"./{data_dir}/" + f"{dataset}/"+ dataset + "_image_embedding_test.npy", features_test)
np.savetxt(f"./{data_dir}/" + f"{dataset}/" + dataset + "_labels_train.txt", labels)
np.savetxt(f"./{data_dir}/" + f"{dataset}/" + dataset +  "_labels_test.txt", labels_test)
print("Saved image features and labels.")