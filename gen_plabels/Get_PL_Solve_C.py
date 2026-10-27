import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import faiss
import numpy as np
import torch.nn.functional as F
from kmeans import kmeans_clustering
from draw_martix import compute_metrics
import matplotlib.pyplot as plt
dataset = "DTD"  # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"], ["EuroSAT"]

if dataset == "CIFAR-10" or dataset == "STL-10"  or dataset == "EuroSAT":
    cluster_num = 10
    over_clustering_num = 167
elif dataset == "Flowers":
    cluster_num = 102
    over_clustering_num = 306
elif dataset == "ImageNet-10":
    cluster_num = 10
    over_clustering_num = 43
elif dataset == "CIFAR-20":
    cluster_num = 20
    over_clustering_num = 167
elif dataset == "ImageNet-Dogs":
    cluster_num = 15
    over_clustering_num = 65
elif dataset == "DTD":
    cluster_num = 47
    over_clustering_num = 141
elif dataset == "UCF101":
    cluster_num = 101
elif dataset == "ImageNet":
    cluster_num = 1000
elif dataset == "TinyImageNet":
    cluster_num = 200
    over_clustering_num = 333
# cluster_num = 15
# over_clustering_num = 65  # [167, 167, 17, 43, 65, 141, 303, 4271], [63]

split = 'train+val' # 'train' # 'train', 'test', 'train+unlabeled'

if split == None:
    dataset_dir = dataset
else:
    dataset_dir = dataset + '-' + split

root_dir = 'data'
backbone = 'b32'

tau = 0.005

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

def fliter_nouns():

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
            f"./gen_plabels/{root_dir}/" + f"{dataset_dir}/" + dataset_dir + "_image_" + str(over_clustering_num) + "cluster.npy"
        )
    except:

        # preds 是 167个聚类中心的结果
        preds, _ = kmeans_clustering(images_embedding, over_clustering_num)
        np.save(
            f"./gen_plabels/{root_dir}/" + f"{dataset_dir}/" + dataset_dir + "_image_" + str(over_clustering_num) + "cluster.npy", preds.cpu().numpy()
        )
        print("Please rerun the script.")
        exit()

    image_centers = torch.zeros((over_clustering_num, 512), dtype=torch.float16).cuda()
    for k in range(over_clustering_num):
        image_centers[k] = images_embedding[preds == k].mean(dim=0)
    image_centers = F.normalize(image_centers, dim=1)

    similarity = torch.matmul(image_centers, nouns_embedding.T)
    softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()
    # 对每个名词，选择与其最相关的簇（即 softmax 分布中概率最大的簇）
    class_pred = torch.argmax(softmax_nouns, dim=0).long()
    # 根据簇的数量，选择每个簇中概率最高的 topK 个名词 ，此处尝试不同的 topK
    
    

    for topK in [1, 2, 3, 4, 5, 8, 10, 20]: #[1, 2, 3, 4, 5, 8, 10, 20]:
        print(f"Top {topK} nouns per cluster")
        selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
        for k in range(over_clustering_num):
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
            "./gen_plabels/extend_nouns_emb/data/" + f"{dataset_dir}/" + dataset_dir + f"_{backbone}_{over_clustering_num}_filtered_nouns_embedding_topk-{topK}.npy",
            nouns_embedding_selected.cpu().numpy(),
        )


def draw(x, title):
    # 示例 tensor（用户运行时可替换为自己的 tensor）
    x_sorted = torch.sort(x, descending=True).values

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(x_sorted)), x_sorted)
    plt.title("Sorted Tensor Values (Descending)")
    plt.xlabel("Index (sorted)")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(title, dpi=300, bbox_inches="tight")



def Solve_C():
    for topK_for_C in [1, 2, 3, 4, 5, 8, 10, 20]:
        
      
        gt_labels = np.loadtxt(f'gen_plabels/data/{dataset_dir}/{dataset_dir}_labels_train.txt')

        # 读取嵌入数据
        X_np = np.load(f'gen_plabels/data/{dataset_dir}/{dataset_dir}_image_embedding_train.npy')
        # W_np = np.load(f'gen_plabels/extend_nouns_emb/data/{dataset_dir}/{dataset_dir}_{backbone}_{over_clustering_num}_filtered_nouns_embedding_topk-{topK_for_C}.npy')
        W_np = np.load(f'gen_plabels/extend_nouns_emb/data/{dataset_dir}/{dataset_dir}_filtered_nouns_embedding_topk-{topK_for_C}.npy')

        # 转换为 PyTorch 张量
        X = torch.tensor(X_np, dtype=torch.float32)  # n x d
        W = torch.tensor(W_np, dtype=torch.float32)  # m x d

        print(f"X shape: {X.shape}")  # 应该是 n x d
        print(f"W shape: {W.shape}")  # 应该是 m x d

        # 确定是test还是save
        test = True

        # 设置正则化参数 gamma
        for gamma in [5]:
            print(f'gamma:{gamma}')

            # 计算最优的 C
            # C = X @ W.T @ inv(gamma * I + W @ W.T)

            # W @ W.T 的形状是 m x m
            WWT = W @ W.T  # m x m

            # gamma * I + W @ W.T
            m = W.shape[0]
            I = torch.eye(m, dtype=torch.float32)
            A = gamma * I + WWT  # m x m

            # 计算逆矩阵
            A_inv = torch.linalg.inv(A)  # m x m

            # X @ W.T 的形状是 n x m
            XWT = X @ W.T  # n x m

            # 最终的 C
            C = XWT @ A_inv  # n x m

            print(f"\nOptimal C shape: {C.shape}")
            print(f"C:\n{C}")

            # 验证目标函数值
            reconstruction_error = torch.norm(X - C @ W, p='fro') ** 2
            regularization = gamma * torch.norm(C, p='fro') ** 2
            total_loss = reconstruction_error + regularization

            print(f"\n=== 目标函数值 ===")
            print(f"重建误差 ||X - CW||_F^2: {reconstruction_error.item():.6f}")
            print(f"正则化项 γ||C||_F^2: {regularization.item():.6f}")
            print(f"总损失: {total_loss.item():.6f}")
            
           
            seed = 0
            print(f'seed:{seed}')
            preds, _ = kmeans_clustering( (C).cuda(), cluster_num, 0)
            
            acc_, nmi_, anmi_, ari_,  = compute_metrics(torch.tensor(gt_labels).to(torch.int).cpu(), 
                                                                            torch.tensor(preds).to(torch.int).cpu(), 
                                                                            min_samples_per_class=5, 
                                                                            return_reassign = False,
                                                                            draw_matrix = False,
                                                                            fname = f'c10-seed{seed}-gamma-{gamma}' 
                                                                            )
            
            print(f"acc-{acc_:.4f} nmi-{nmi_:.4f} ari-{ari_:.4f}")

            if test:        
                with open(f"output_self/diff/albation/{dataset_dir}.txt", "a") as f:
                    # f.write(f'gamma:{gamma}\n')
                    f.write(f'topk:{topK_for_C} seed:{seed} gamma:{gamma}\n')
                    # for seed, acc, nmi, ari in zip(seeds, acc_list, nmi_list, ari_list):
                    f.write(
                        f"ACC={acc_:.4f}, "
                        f"NMI={nmi_:.4f}, "
                        f"ARI={ari_:.4f}\n"
                    )

            else:
                save_path = f'plabels/{dataset_dir}-{backbone}/C_topk{topK_for_C}-gamma{gamma}-seed{seed}/'
                os.makedirs(save_path, exist_ok=True)
                torch.save(torch.tensor(preds).to(torch.int), save_path + 'pl.pt' )
                torch.save(C, save_path + 'C.pt' )

    

if __name__ == "__main__":
    # 1) 过滤名词
    # fliter_nouns()

    # 2) 计算相似度并保存
    # compute_sim()

    # 3) 求解 C 矩阵并聚类
    Solve_C()