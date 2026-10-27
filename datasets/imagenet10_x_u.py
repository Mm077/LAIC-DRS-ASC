import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

import torch

@DATASET_REGISTRY.register()
class ImageNet10_X_U(DatasetBase):

    # dataset_dir = "imagenet"

    def __init__(self, cfg):

        self.dataset_dir = '/nas/datasets/imagenet/'
        self.image_dir = os.path.join(self.dataset_dir, "images")

        # ✅ 只使用 ImageNet-10 的类定义文件
        self.subset_file = './datasets/imagenet10_classes.txt'

        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed_imagenet10.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            # ✅ 只读取 ImageNet-10 的 classnames
            classnames = self.read_classnames(self.subset_file)

            train = self.read_data(classnames, "train")
            test = self.read_data(classnames, "train")

            preprocessed = {"train": train, "test": test}
            # 保存预处理的数据以加快后续加载速度 暂时不用
            # with open(self.preprocessed, "wb") as f:
            #     pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # ✅ 可选的 further 子类采样（可保持不变）
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        train_x, train_u = self.generate_Train_X_U(train)
        super().__init__(train_x=train_x, train_u=train_u, val=test, test=test)

    # ----------------------------------------------------
    # ✅ 读取 ImageNet-10 的类别定义
    # ----------------------------------------------------
    @staticmethod
    def read_classnames(text_file):
        """
        Return:
            OrderedDict: {folder_name: class_name}
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    # ----------------------------------------------------
    # ✅ 只加载 ImageNet-10 对应子文件夹的数据
    # ----------------------------------------------------
    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.dataset_dir, split_dir)

        # ✅ 关键：仅保留 subset10 中定义的 10 个类
        folders = sorted(
            f.name for f in os.scandir(split_dir)
            if f.is_dir() and f.name in classnames
        )

        items = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]

            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)

        print(
            f"Loaded ImageNet-10 | split={split_dir} | "
            f"num_classes={len(folders)} | num_images={len(items)}"
        )

        return items
    
    def generate_Train_X_U(self, train):
 
        indexs = torch.tensor(torch.load('plabels/ImageNet-10-b32/C_topk2-gamma5-seed5_extract/pidx_fromknn-10-th1.pt')).cpu().flatten()
        plabels = torch.tensor(torch.load('plabels/ImageNet-10-b32/C_topk2-gamma5-seed5_extract/pl_fromknn-10-th1.pt')).cpu().flatten()

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
            # else:
                # 无标签样本
            new_item = Datum(impath=item.impath, label=1, classname=item.classname, index=index)
            items_ulb.append(new_item)

        return items_lb, items_ulb
