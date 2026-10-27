import os
import numpy as np
from tqdm import tqdm

import torch
from torch.backends import cudnn


from utils.draw_confusion import compute_metrics, draw_confusion, draw_p_gt_matrix
from utils.kmeans import kmeans_clustering

# from ensemble_clustering import en_clustering_on_diff_topk

k = 100


# 余弦相似度
@torch.no_grad()
def compute_neighbors(embedding):
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    num_embeds = embedding.shape[0]
    if num_embeds <= 8 * 1e4:
        dists = embedding @ embedding.permute(1, 0)
        # exclude self-similarity
        dists.fill_diagonal_(-torch.inf)
        return dists.topk(k, dim=-1)
    else:
        topk_knn_ids = []
        topk_knn_dists = []
        print("Chunk-wise implementation of k-nn in GPU")
        # num_chunks = 12000
        step_size = 64  # num_embeds // num_chunks
        embedding = embedding.cuda()
        for idx in tqdm(range(0, num_embeds, step_size)):
            idx_next_chunk = min((idx + step_size), num_embeds)
            features = embedding[idx: idx_next_chunk, :]
            # calculate the dot product dist
            dists_chunk = torch.mm(features, embedding.T).cpu()
            dists_chunk.fill_diagonal_(-torch.inf)
            max_dists, indices = dists_chunk.topk(k, dim=-1)
            topk_knn_ids.append(indices)
            topk_knn_dists.append(max_dists)
        return torch.cat(topk_knn_dists), torch.cat(topk_knn_ids)

def get_nn(train_embeddings):

    nn_dists, neighbors = compute_neighbors(train_embeddings)
    return nn_dists, neighbors

def get_knn_acc(neighbors, labels, topks):

    result = dict()
    for topk in topks:
        same_num = 0
        print(f"=> eval top{topk} acc (gt label): ")
        for iter, neighbor in enumerate(neighbors):
            anchor_label = labels[iter]
            topk_neighbor_labels = labels[neighbor[:topk].cpu()]
            same_num += (anchor_label == topk_neighbor_labels).sum()
        topk_acc = same_num / (topk * neighbors.shape[0])
        result[topk] = topk_acc
        print(f"top{topk} acc: {topk_acc}")
    return result

def get_clean_embeddings(neighbors, labels, threshold, topks):
    result = dict()
    for topk in topks:
        clean_indexes = []
        print(f"=> select clean indexes with topk={topk} threshold={threshold} ")
        for iter, neighbor in enumerate(neighbors):
            anchor_label = labels[iter]
            topk_neighbor_labels = labels[neighbor[:topk].cpu()]
            acc = (anchor_label == topk_neighbor_labels).sum() / topk
            if acc >= threshold:
                clean_indexes.append(torch.tensor(iter))
        result[topk] = torch.stack(clean_indexes,dim=0)
    return result

def eval_clean_embeddings(clean_indexes, pseudo_labels, gt_labels, reassign):
    right_num = 0
    # Switch to dict
    # reassign的含义：一个个映射，[对应的伪标签：真实标签]
    reassign_dict = dict()
    for i in reassign:
        reassign_dict[i[1]] = i[0]
    for index in clean_indexes:
        if reassign_dict[gt_labels[index].item()] == pseudo_labels.cpu()[index.cpu()].item():
            right_num += 1

    acc = right_num / clean_indexes.shape[0]
    return acc

def eval_clean_embeddings_by_each_class(clean_indexes, pseudo_labels, gt_labels, reassign):
    reassign_dict = dict()
    for i in reassign:
        reassign_dict[i[1]] = i[0]

    results_acc_each_class = dict()

    n_cls = torch.unique(pseudo_labels).shape[0]
    for i in range(n_cls):
        results_acc_each_class[i] = 0
    pseudo_labels = pseudo_labels.cpu()
    for index in clean_indexes:
        if reassign_dict[gt_labels[index].item()] == pseudo_labels[index].item():
            results_acc_each_class[pseudo_labels[index].item()] +=1
    return results_acc_each_class

def fill_missing_labels_torch(pseudo_labels, num_classes=20, seed=42):
    """
    pseudo_labels: 1D torch.Tensor of shape [N], int64 type, values in 0~19
    每个缺失类从已有样本中随机挑选2个样本改为该类
    """
    torch.manual_seed(seed)

    unique_labels = torch.unique(pseudo_labels).tolist()
    full_labels = set(range(num_classes))
    missing_labels = list(full_labels - set(unique_labels))

    print(f"缺失标签: {missing_labels}")

    # 所有非缺失类别的索引
    valid_mask = torch.tensor([lbl not in missing_labels for lbl in pseudo_labels], dtype=torch.bool)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

    for missing_label in missing_labels:
        if valid_indices.numel() < 2:
            print(f"剩余可用样本不足 2 个，无法补足缺失标签 {missing_label}")
            break

        # 随机选两个不同的索引
        selected = valid_indices[torch.randperm(valid_indices.numel())[:2]]

        for idx in selected:
            print(f"将索引 {idx.item()} 的标签 {pseudo_labels[idx].item()} 替换为缺失标签 {missing_label}")
            pseudo_labels[idx] = missing_label

        # 移除已使用的索引，防止重复替换
        valid_indices = valid_indices[~torch.isin(valid_indices, selected)]

    return pseudo_labels

def eval_clean_embeddings_by_each_class_returnN(clean_indexes, pseudo_labels, gt_labels, reassign):
    """
    评估每个伪标签类别在 clean_indexes 中的预测准确性。
    reassign 是一个伪标签 → 真实标签的映射列表。

    Args:
        clean_indexes (List[int]): 被选中的干净样本索引
        pseudo_labels (Tensor): [N]，每个样本的伪标签
        gt_labels (Tensor): [N]，每个样本的真实标签
        reassign (List[List[int]]): 每一项是 [pseudo_label, gt_label]

    Returns:
        dict[int, dict]: 每个伪标签类的 {"correct": x, "total": y, "acc": z}
    """
    # ✅ 改动：建立集合，便于判断 [pseudo, gt] 是否匹配
    reassign_set = set((p, g) for p, g in reassign)

    results_acc_each_class = dict()
    n_cls = torch.unique(pseudo_labels).shape[0]

    for i in range(n_cls):
        results_acc_each_class[i] = {
            "correct": 0,
            "total": 0,
            "acc": 0.0
        }

    pseudo_labels = pseudo_labels.cpu()
    gt_labels = gt_labels.cpu()

    for index in clean_indexes:
        pseudo = pseudo_labels[index].item()
        gt = gt_labels[index].item()

        if (pseudo, gt) in reassign_set:  # ✅ 改动：判断 [pseudo → gt] 是否符合映射
            results_acc_each_class[pseudo]["correct"] += 1
        results_acc_each_class[pseudo]["total"] += 1

    for cls_id, stats in results_acc_each_class.items():
        total = stats["total"]
        correct = stats["correct"]
        stats["acc"] = correct / total if total > 0 else 0.0

def extract_pseudo_labels_knn_on_embs(cfg, topk):
    cudnn.benchmark = True
    cudnn.deterministic = True
    

    # => 加载gt标签
    dataset_pl_dir = 'ImageNet-Dogs'      # 'CIFAR-20-train' 'DTD-all' 'DTD-train+val' 
    cfg.cluster_num = 102
    backbone = 'b32'
    over_clus_num = 167
    plname = f'C_topk{topk}-gamma5-seed5'


    gt_labels_path = f'./plabels/{dataset_pl_dir}-{backbone}/labels_train.txt'
    gt_labels = torch.tensor(np.loadtxt(gt_labels_path), dtype=torch.int64)

    train_embeddings_path = f'/plabels/{dataset_pl_dir}-{backbone}/train.pt'
    train_embeddings = torch.tensor(torch.load(train_embeddings_path), dtype=torch.float32)


    # => kmeans计算伪标签 或者加载伪标签
    print(f"kmeans cls num:{cfg.cluster_num}")
    # pseudo_labels_faiss = kmeans(fea, cfg.cluster_num)
    compute_pl = False
    if compute_pl:
        pseudo_labels, centers = kmeans_clustering(train_embeddings, cfg.cluster_num)
    
    else:
      
        pseudo_labels = torch.load(f'plabels/{dataset_pl_dir}-{backbone}/{plname}/pl.pt')
      

    # reassign的含义：一个个映射，[伪标签：对应的真实标签] { [label_from_pseudo, label_from_gt ] ,..., }
    print("propos-kmeans 伪标签质量：")
    acc_, nmi_, anmi_, ari_, reassign = compute_metrics(gt_labels.cpu(), 
                                                        pseudo_labels.cpu(),  
                                                        min_samples_per_class=5, 
                                                        return_reassign = True,
                                                        draw_matrix = False, 
                                                        fname=f'./plabels/cifar20-grad-from-zsclip/')
    print('=> eval results:')
    # print('acc:{} nmi:{} ari:{}'.format(acc_, nmi_, ari_))

    # => 获取近邻
    nn_dists, neighbors = compute_neighbors(train_embeddings)

    topks = [1, 5, 10, 20, 30, 40, 50] # [60, 70, 80, 90, 100]
    # topks = [15, 25, 35, 45] # [60, 70, 80, 90, 100]
    thresholds = [1] # [0.9, 0.92, 0.95, 0.99]

    # 伪标签存储地址

    save_path = f'./plabels/{dataset_pl_dir}-{backbone}/{plname}_extract/'

    os.makedirs(save_path, exist_ok=True)
    
    
    for threshold in thresholds:
        print(f"================== Now threshold = {threshold} ==================")
        # => 测试近邻的准确率

        # => 此处：得到干净伪标签idx ， clean_indexes是一个字典，包含不同topk下的筛选出的干净标签。clean_indexes[topk] = [id1, id2, ..., idn]
        clean_indexes = get_clean_embeddings(neighbors, pseudo_labels, threshold, topks)

        # => 测量得到的干净伪标签的准确率，直接经过映射反转之后和真实标签进行比对，不再进行匈牙利匹配
        writer = []

        for topk in topks:
            p_num = clean_indexes[topk].shape[0]
            acc = eval_clean_embeddings(clean_indexes[topk], pseudo_labels, gt_labels, reassign)
            print(f"topk={topk}: acc={acc}")
      
            pl = fill_missing_labels_torch(pseudo_labels[clean_indexes[topk]], num_classes=cfg.cluster_num)
            n, num = torch.unique(pl, return_counts=True)
            
            print(f"伪标签数量：{p_num}，每个标签的样本数：{num}")

            torch.save(clean_indexes[topk], f'{save_path}' + f'pidx_fromknn-{topk}-th{threshold}.pt')
            torch.save(pl, f'{save_path}' + f'pl_fromknn-{topk}-th{threshold}.pt')
            writer.append(f"topk={topk}: acc={acc}, p_num={p_num}, threshold={threshold}\n")

    np.savetxt(save_path + "writer2.txt", writer, fmt="%s")
    print("finished")

def main():
    from easydict import EasyDict as edict

    __C = edict()
    cfg = __C
    ## Set training options
    __C.dataset = 'CIFAR-20' # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
    __C.batch_size = 256
    __C.epochs = 1000
    __C.cluster_num = 20
    __C.embed_dim = 32
    __C.heads_num = 8
    __C.load = False 
    # for topk in [1,2,3,4,5,8,10,20]:
    for topk in [2]:
        extract_pseudo_labels_knn_on_embs(cfg, topk)

if __name__ == '__main__':

    main()





