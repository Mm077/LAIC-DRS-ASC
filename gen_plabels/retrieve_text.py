import torch
import numpy as np
import torch.nn.functional as F

dataset = "CIFAR-20"  # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"],["EuroSAT"]
tau = 0.005

root_dir = 'data'
backbone = 'b32'
over_clustering_num = 200

split = None # 'train', 'test', 'train+unlabeled'
if split == None:
    dataset_dir = dataset
else:
    dataset_dir = dataset + '-' + split

for topk in [1, 2, 3, 4, 5, 8, 10, 20]:  # [1, 3, 5, 10]

    print(f"topk: {topk}")
    nouns_embedding = np.load("./gen_plabels/extend_nouns_emb/data/" + f"{dataset_dir}/" + dataset_dir + f"_{backbone}_{over_clustering_num}_filtered_nouns_embedding_topk-{topk}.npy")
    # nouns_embedding = np.load("./extend_nouns_emb/data/" + f"{dataset}/" + f"filtered_var_nouns_embedding_topk-{topk}.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )
    images_embedding = np.load(f"./gen_plabels/{root_dir}/" + f"{dataset_dir}/" + dataset_dir + "_image_embedding_train.npy")
    images_embedding = images_embedding / np.linalg.norm(
        images_embedding, axis=1, keepdims=True
    )

    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]
    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]

    retrieval_embeddings = []
    similarity_all = []
    batch_size = 8192
    for i in range(image_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > image_num:
            end = image_num
            images_batch = images_embedding[start:end]
        similarity = torch.matmul(images_embedding[start:end], nouns_embedding.T)
        # similarity = torch.softmax(similarity / tau, dim=1)
        similarity_all.append(similarity.cpu().float())
        retrieval_embedding = (similarity @ nouns_embedding).cpu()
        retrieval_embeddings.append(retrieval_embedding)
        if i % 50 == 0:
            print(f"[Completed {i * batch_size}/{image_num}]")
    similarity_all = torch.cat(similarity_all, dim=0)
    torch.save(similarity_all, "./gen_plabels/extend_nouns_emb/data/"+ f"{dataset_dir}/similarity_wosoftmax_{backbone}_{over_clustering_num}_topk-{topk}.pt")
