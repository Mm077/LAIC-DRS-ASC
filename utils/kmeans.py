import torch_clustering

def kmeans_clustering(features, n_clusters):
    kwargs = {
        'metric': 'cosine',  # if self.l2_normalize else 'euclidean',
        'distributed': True,
        'random_state': 0,
        'n_clusters': n_clusters,
        'verbose': True
    }
    clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
    
    psedo_labels = clustering_model.fit_predict(features)
    cluster_centers = clustering_model.cluster_centers_
    return psedo_labels, cluster_centers
