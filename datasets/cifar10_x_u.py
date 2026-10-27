import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets  # 用于 split 和 subsample，dassl 中已有工具

import torch
@DATASET_REGISTRY.register()
class CIFAR10_X_U(DatasetBase):

    # train split = 50000
    dataset_dir = "cifar10_all=False"
    
    # train split = 60000
    dataset_dir_all = "cifar10"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.dataset_dir_all = os.path.join(root, self.dataset_dir_all)
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        # 这里，使用所有样本用于测试：
        # self.test_dir = os.path.join(self.dataset_dir_all, "train")

        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 类名与标签
        classnames = sorted(os.listdir(self.train_dir))
        cname2lab = {c: i for i, c in enumerate(classnames)}
        print(f"Class names: {cname2lab}")

        if cfg.RUN_TYPE == "GEN_EMB" or cfg.RUN_TYPE == "EVAL_PRED":
            # 仅用于生成特征或评估预测
            train, _ = self.read_data(self.train_dir, cname2lab) # , split_train = True)
        else:
            # 用于训练
            train_x, train_u = self.read_data(self.train_dir, cname2lab, split_train = True)

        test, _ = self.read_data(self.test_dir, cname2lab)
        _, val = self.split_trainval(train_x, val_ratio=0.1)


        # #2025年6月12日18:04:29

        # few-shot 处理
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            load_preprocess = False
            if os.path.exists(preprocessed) and load_preprocess:
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train_x, val = data["train"], data["val"]
            else:
                train_x = self.generate_fewshot_dataset(train_x, num_shots=num_shots)
                # val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train_x, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # 可选：类子集抽样
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train_x, val, test = OxfordPets.subsample_classes(train_x, val, test, subsample=subsample)

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def read_data(self, dir_path, cname2lab, split_train = False):

                    
        indexs = torch.tensor(torch.load('plabels/CIFAR-10-train-b32/C_topk2-gamma5-seed20_extract/pidx_fromknn-10-th1.pt')).cpu().flatten()
        plabels = torch.tensor(torch.load('plabels/CIFAR-10-train-b32/C_topk2-gamma5-seed20_extract/pl_fromknn-10-th1.pt')).cpu().flatten()
        
        # 创建一个索引到伪标签的映射
        index_to_pl_pseudo = index_to_pseudo = {int(idx.item()): int(plabels[i].item()) for i, idx in enumerate(indexs)}

        items_lb = []
        items_ulb = []
        for classname in os.listdir(dir_path):
            label = cname2lab[classname]
            class_dir = os.path.join(dir_path, classname)
            for fname in os.listdir(class_dir):
                if not fname.endswith(".png"):
                    continue  # 跳过非 png 文件
                impath = os.path.join(class_dir, fname)

                index_str = os.path.splitext(fname)[0]
                try:
                    index = int(index_str)  # 如果希望是 int 类型
                except ValueError:
                    index = index_str  # 如果不保证是数字，可保留字符串
                
                if split_train:
                    # split_train 为 True 时，意思为用于训练阶段，其label加载是从index_to_pseudo中加载的伪标签
                    if index in index_to_pseudo:   # 说明是有标记样本，导入items_lb:
                        p_label = index_to_pseudo[index]
                        pl_label = index_to_pl_pseudo[index]
                        item = Datum(impath=impath, label=p_label, pl_label=pl_label, classname=classname, index=index)
                        items_lb.append(item)
                    
                    else:
                        item = Datum(impath=impath, label=0, classname=classname, index=index)
                        items_ulb.append(item)

                else:
                    item = Datum(impath=impath, label=label, classname=classname, index=index)
                    items_lb.append(item) 

        return items_lb, items_ulb

        
       
       
       

    def split_trainval(self, data, val_ratio=0.1):
        from sklearn.model_selection import train_test_split
        train, val = train_test_split(data, test_size=val_ratio, stratify=[d.label for d in data])
        return train, val

