import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import faiss
import numpy as np
import torch.nn.functional as F


def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    X = X.astype(np.float32)
    kmeans = faiss.Kmeans(d, cluster_num, gpu=False, spherical=True, niter=300, nredo=10)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I


if __name__ == "__main__":
    dataset = "CIFAR-20"  # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"], ["EuroSAT"]
    cluster_num = 200  # [167, 167, 17, 43, 65, 141, 303, 4271], [63]
    topK = 5
    split = None # 'train', 'test', 'train+unlabeled'
    
    if split == None:
        dataset_dir = dataset
    else:
        dataset_dir = dataset + '-' + split

    root_dir = 'data'
    backbone = 'b32'

    # nouns_embedding = np.load("./extend_nouns_emb/data/CIFAR-20/nouns_embedding_ensemble.npy")
    nouns_embedding = np.load(f"./gen_plabels/{root_dir}/nouns_embedding_ensemble.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )
    images_embedding = np.load(f"./gen_plabels/{root_dir}/" + f"{dataset_dir}/" + dataset_dir +  "_image_embedding_train.npy")
    images_embedding = images_embedding / np.linalg.norm(
        images_embedding, axis=1, keepdims=True
    )

    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]

    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]

    try:
        preds = np.load(
            f"./gen_plabels/{root_dir}/" + f"{dataset_dir}/" + dataset_dir + "_image_" + str(cluster_num) + "cluster.npy"
        )
    except:

        # preds 是 167个聚类中心的结果
        preds = kmeans(images_embedding.cpu().numpy(), cluster_num)
        np.save(
            f"./gen_plabels/{root_dir}/" + f"{dataset_dir}/" + dataset_dir + "_image_" + str(cluster_num) + "cluster.npy", preds
        )
        print("Please rerun the script.")
        exit()

    image_centers = torch.zeros((cluster_num, 512), dtype=torch.float16).cuda()
    for k in range(cluster_num):
        image_centers[k] = images_embedding[preds == k].mean(dim=0)
    image_centers = F.normalize(image_centers, dim=1)

    similarity = torch.matmul(image_centers, nouns_embedding.T)
    softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()
    # 对每个名词，选择与其最相关的簇（即 softmax 分布中概率最大的簇）
    class_pred = torch.argmax(softmax_nouns, dim=0).long()
    # 根据簇的数量，选择每个簇中概率最高的 topK 个名词 ，此处尝试不同的 topK
    
    
    for topK in [1, 2, 3, 4, 5, 8, 10, 20]:
        print(f"Top {topK} nouns per cluster")
        selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
        for k in range(cluster_num):
            if (class_pred == k).sum() == 0:
                continue
            class_index = torch.where(class_pred == k)[0]
            softmax_class = softmax_nouns[:, class_index]
            confidence = softmax_class.max(dim=0)[0]
            rank = torch.argsort(confidence, descending=True)
            selected_idx[class_index[rank[:topK]]] = True
        selected_idx = selected_idx.cpu().numpy()

        print(selected_idx.sum(), "nouns selected.")
        nouns_embedding_selected = nouns_embedding[selected_idx]

        np.save(
            # "./extend_nouns_emb/data/" + f"{dataset}/" + dataset + f"_filtered_nouns_embedding_topk-{topK}.npy",
            "./gen_plabels/extend_nouns_emb/data/" + f"{dataset_dir}/" + dataset_dir + f"_{backbone}_{cluster_num}_filtered_nouns_embedding_topk-{topK}.npy",
            nouns_embedding_selected.cpu().numpy(),
        )
