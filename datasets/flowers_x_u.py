import os
import pickle
import random
import json
from typing import Callable, Optional, Tuple, Any

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing
from .oxford_pets import OxfordPets
import torch

@DATASET_REGISTRY.register()
class Flowers_X_U(DatasetBase):

    dataset_dir = "flowers-102"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = '/nas/datasets/'  # 数据集路径
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")  # 使用split文件
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 加载划分数据
        if os.path.exists(self.split_path):
            print(f"Loading split file from {self.split_path}")
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
            with open(self.split_path, "r") as f:
                split_data = json.load(f)
        else:
            print(f"Split file not found. Generating new split...")
            split_data = self.read_and_split_data(self.image_dir)
            with open(self.split_path, "w") as f:
                json.dump(split_data, f, indent=4)
            print(f"Saved split file to {self.split_path}")

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
        
        # 根据 idx 划分为有标签和无标签两部分
        train_x, train_u = self.generate_Train_X_U(train + val)
        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        # 读取并划分数据
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test

    def generate_Train_X_U(self, train):
        # 加载伪标签（plabels）及其对应索引
        indexs = torch.tensor(torch.load('plabels/Flowers-train+val-b32/C_topk2-gamma5-seed20_extract/pidx_fromknn-1-th1.pt')).cpu().flatten()
        plabels = torch.tensor(torch.load('plabels/Flowers-train+val-b32/C_topk2-gamma5-seed20_extract/pl_fromknn-1-th1.pt')).cpu().flatten()

        # 创建一个索引到伪标签的映射
        index_to_pseudo = {int(idx.item()): int(plabels[i].item()) for i, idx in enumerate(indexs)}

        items_lb = []
        items_ulb = []
        for index, item in enumerate(train):
            # 有标签样本
            if index in index_to_pseudo:
                plabel = index_to_pseudo[index]
                new_item = Datum(impath=item.impath, label=plabel, pl_label=plabel, classname=item.classname, index=index)
                items_lb.append(new_item)
            # else:
                # 无标签样本
            new_item = Datum(impath=item.impath, label=1, classname=item.classname, index=index)
            items_ulb.append(new_item)

        return items_lb, items_ulb

    def extra_repr(self) -> str:
        return f"Split: {self.split} | Classes: {len(self.classes)} | Samples: {len(self.samples)}"
