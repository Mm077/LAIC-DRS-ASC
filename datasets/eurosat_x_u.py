import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

import torch

NEW_CNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}


@DATASET_REGISTRY.register()
class EuroSAT_X_U(DatasetBase):

    dataset_dir = "EuroSAT-all"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = '/nas/datasets/'
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        
        train_x, train_u = self.generate_Train_X_U(train+val, test)

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    
    def generate_Train_X_U(self, train, test):

            # 加载pl及其idx
            # indexs = torch.tensor(torch.load('plabels/UCF101-all/knn:fea=CONCAT-pl=kmeansonCONCAT/pidx_fromknn-1-th1.pt')).cpu().flatten()
            # plabels = torch.tensor(torch.load('plabels/UCF101-all/knn:fea=CONCAT-pl=kmeansonCONCAT/pl_fromknn-1-th1.pt')).cpu().flatten()
            
            indexs = torch.tensor(torch.load('plabels/EuroSAT/knn:fea=CONCAT-pl=SIMTop5/pidx_fromknn-1-th1.pt')).cpu().flatten()
            plabels = torch.tensor(torch.load('plabels/EuroSAT/knn:fea=CONCAT-pl=SIMTop5/pl_fromknn-1-th1.pt')).cpu().flatten()

            # 创建一个索引到伪标签的映射
            index_to_pseudo = {int(idx.item()): int(plabels[i].item()) for i, idx in enumerate(indexs)}

            items_lb = []
            items_ulb = []
            for index, item in enumerate(train):
                # 有标签样本
                if index in index_to_pseudo:
                    label = index_to_pseudo[index]
                    # 这里没有给正确的pllabel
                    new_item = Datum(impath=item.impath, label=label, pl_label=label, classname=item.classname, index=index)
                    items_lb.append(new_item)
                else:
                    # 无标签样本
                    new_item = Datum(impath=item.impath, label=1, classname=item.classname, index=index)
                    items_ulb.append(new_item)

            # for index, item in enumerate(test):
            #     new_item = Datum(impath=item.impath, label=1, classname=item.classname, index=index + len(train))
            #     items_ulb.append(new_item)

            return items_lb, items_ulb
