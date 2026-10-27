import faiss
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder # import CIFAR10, CIFAR100,
from tools.datasets import DTD, UCF101, EuroSAT, OxfordFlowers
from tools.datasets import CIFAR10, CIFAR100, STL10
from tools.in_datasets import ImageNetSubEmb, TinyImageNetDataset
BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(dataset="CIFAR-10"):
    if (
        dataset == "CIFAR-10"
        or dataset == "CIFAR-20"
        or dataset == "STL-10"
        or dataset == "DTD"
        or dataset == "UCF101"
        or dataset == "EuroSAT"
        or dataset == "Flowers"
    ):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif (
        dataset == "ImageNet-Dogs" or dataset == "ImageNet-10" or dataset == "ImageNet" or dataset == "TinyImageNet"
    ):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                    # (0.48145466, 0.4578275, 0.40821073),
                    # (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        raise NotImplementedError
    return transforms


def get_dataloader(dataset="CIFAR-10", batch_size=4096):
    transforms = get_transforms(dataset)
    if dataset == "CIFAR-10":
        data_train = CIFAR10(
            root="/nas/datasets/", train=True, download=True, transform=transforms, all=True
        )
        data_test = CIFAR10(
            root="/nas/datasets/", train=False, download=True, transform=transforms
        )
    elif dataset == "CIFAR-20":
        data_train = CIFAR100(
            root="/nas/datasets/", train=True, download=True, transform=transforms, all=True
        )
        data_test = CIFAR100(
            root="/nas/datasets/", train=False, download=True, transform=transforms
        )
    elif dataset == "STL-10":
        data_train = STL10(
            root="/nas/datasets/", split="train+test", download=True, transform=transforms
        )
        data_test = STL10(
            root="/nas/datasets/", split="test", download=True, transform=transforms
        )
    elif dataset == "ImageNet-10":
        data_train = ImageNetSubEmb(            
            subset_file= './gen_plabels/data/ImageNet-10/imagenet10_classes.txt', embedding=None, split='train', transform1=transforms
        )

        data_test = ImageNetSubEmb(            
            subset_file= './gen_plabels/data/ImageNet-10/imagenet10_classes.txt', embedding=None, split='val', transform1=transforms
        )   

        # data_train = ImageFolder("./data/ImageNet-10/train", transform=transforms)  data/ImageNet-Dogs/imagenet_dogs_classes.txt
        # data_test = ImageFolder("./data/ImageNet-10/val", transform=transforms)
    elif dataset == "ImageNet-Dogs":
        data_train = ImageNetSubEmb(            
            subset_file= './gen_plabels/data/ImageNet-Dogs/imagenet_dogs_classes.txt', embedding=None, split='train', transform1=transforms
        )

        data_test = ImageNetSubEmb(            
            subset_file= './gen_plabels/data/ImageNet-Dogs/imagenet_dogs_classes.txt', embedding=None, split='val', transform1=transforms
        )   

        # data_train = ImageFolder("./data/ImageNet-Dogs/train", transform=transforms)
        # data_test = ImageFolder("./data/ImageNet-Dogs/val", transform=transforms)
    
    elif dataset == "TinyImageNet":
        data_train = TinyImageNetDataset(
            root_dir='/nas/datasets/tiny-imagenet-200', dataset_type='train', transform=transforms
            )
        # data_train = TinyImageNetDataset(
        #     root="./data/TinyImageNet", split="train", transform=transforms
        # )
        data_test = TinyImageNetDataset(
            root_dir='/nas/datasets/tiny-imagenet-200', dataset_type='val', transform=transforms
        )
    elif dataset == "DTD":
        data_train = DTD(
            root="/nas/datasets/dtd/", split="train+val+test", download=False, transform=transforms
        )
        data_test = DTD(
            root="/nas/datasets/dtd/", split="test", download=False, transform=transforms
        )
        # data_train = ImageFolder("./data/DTD/trainval", transform=transforms)
        # data_test = ImageFolder("./data/DTD/test", transform=transforms)
    
    elif dataset == "UCF101":
        data_train = UCF101(
            # root="/nas/datasets/", split="train+val", download=False, transform=transforms
            root="/nas/datasets/", split="all", download=False, transform=transforms
        )
        data_test = UCF101(
            root="/nas/datasets/", split="test", download=False, transform=transforms
        )
        # data_train = ImageFolder("./data/UCF101/train", transform=transforms)
        # data_test = ImageFolder("./data/UCF101/val", transform=transforms)

    elif dataset == "EuroSAT":
        data_train = EuroSAT(
            root="/nas/datasets/", split="train+val", download=False, transform=transforms
        )
        data_test = EuroSAT(
            root="/nas/datasets/", split="test", download=False, transform=transforms
        )
    
    elif dataset == "Flowers":
        data_train = OxfordFlowers(root = "/nas/datasets/", split = "train+val", download=False, transform=transforms)
        data_test = OxfordFlowers(root = "/nas/datasets/", split = "test", download=False, transform=transforms)
    # elif dataset == "ImageNet-10":
    #     data_train = ImageNetSubEmb(            
    #         subset_file= './data/ImageNet-10/imagenet10_classes.txt', embedding=None, split='train', transform1=transforms
    #     )
                    
    elif dataset == "ImageNet":
        data_train = ImageFolder("./data/ImageNet/train", transform=transforms)
        data_test = ImageFolder("./data/ImageNet/val", transform=transforms)
    else:
        raise NotImplementedError

    dataloader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=False, drop_last=False
    )
    dataloader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return dataloader_train, dataloader_test


def mine_nearest_neighbors_(features, topk=50):
    print("Computing nearest neighbors...")
    features = features.astype(np.float32)
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included
    print("Nearest neighbors computed.")
    return indices[:, 1:]

import faiss
import numpy as np

def mine_nearest_neighbors(features, topk=50, batch_size=128):


    print("Computing nearest neighbors...")
    features = features.astype(np.float32)
    n, dim = features.shape

    # 构建GPU索引
    index = faiss.IndexFlatIP(dim)          # 内积（归一化后=cosine）
    # index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)                     # 整个库仍一次性放进GPU，通常能放下

    # 分批搜索
    all_idx = []
    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        _, idx = index.search(features[b0:b1], topk + 1)  # 含自身
        all_idx.append(idx[:, 1:])          # 去掉自身
    indices = np.vstack(all_idx)

    print("Nearest neighbors computed.")
    return indices



class NeighborsDataset(Dataset):
    def __init__(self, dataset_text, dataset_image, indices_text, indices_image):
        super(NeighborsDataset, self).__init__()

        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.indices_text = indices_text
        self.indices_image = indices_image
        assert self.indices_text.shape[0] == len(self.indices_text)
        assert self.indices_image.shape[0] == len(self.indices_image)

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)
        anchor_image = self.dataset_image.__getitem__(index)
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)

        return anchor_text, anchor_image, neighbor_text, neighbor_image
