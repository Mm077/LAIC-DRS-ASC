import os
import pickle
import re

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets

import torch

@DATASET_REGISTRY.register()
class UCF101_X_U(DatasetBase):

    dataset_dir = "ucf101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = '/nas/datasets/'
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            cname2lab = {}
            filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    label = int(label) - 1  # conver to 0-based index
                    cname2lab[classname] = label

            trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
            test = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
            train, val = OxfordPets.split_trainval(trainval)
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

        train_x, train_u = self.generate_Train_X_U(train+val)
        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(impath=impath, label=label, classname=renamed_action)
                items.append(item)

        return items
    
    def generate_Train_X_U(self, train):

        # 加载pl及其idx
        # indexs = torch.tensor(torch.load('plabels/UCF101-all/knn:fea=CONCAT-pl=kmeansonCONCAT/pidx_fromknn-1-th1.pt')).cpu().flatten()
        # plabels = torch.tensor(torch.load('plabels/UCF101-all/knn:fea=CONCAT-pl=kmeansonCONCAT/pl_fromknn-1-th1.pt')).cpu().flatten()
        
        indexs = torch.tensor(torch.load('plabels/UCF101-train+val-b32/C_topk2-gamma5-seed60_extract/pidx_fromknn-1-th1.pt')).cpu().flatten()
        plabels = torch.tensor(torch.load('plabels/UCF101-train+val-b32/C_topk2-gamma5-seed60_extract/pl_fromknn-1-th1.pt')).cpu().flatten()

        # 创建一个索引到伪标签的映射
        index_to_pseudo = {int(idx.item()): int(plabels[i].item()) for i, idx in enumerate(indexs)}

        items_lb = []
        items_ulb = []
        for index, item in enumerate(train):
            # 有标签样本
            if index in index_to_pseudo:
                p_label = index_to_pseudo[index]
                # 这里没有给正确的pllabel
                new_item = Datum(impath=item.impath, label=p_label, pl_label=p_label, classname=item.classname, index=index)
                items_lb.append(new_item)
            else:
                # 无标签样本
                new_item = Datum(impath=item.impath, label=1, classname=item.classname, index=index)
                items_ulb.append(new_item)

        # for index, item in enumerate(test):
        #     new_item = Datum(impath=item.impath, label=1, classname=item.classname, index=index + len(train))
        #     items_ulb.append(new_item)

        return items_lb, items_ulb
