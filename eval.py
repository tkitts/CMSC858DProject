import pickle

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance, entropy
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import rand_score
import os


def compute_dist_matrix(embeddings, metric='euclidean'):
    return pdist(embeddings, metric=metric)


def distance_distortion(gt_embeddings, new_embeddings):
    gt_dist_matrix = compute_dist_matrix(gt_embeddings)
    gt_dist_matrix /= np.max(gt_dist_matrix)
    dist_matrix = compute_dist_matrix(new_embeddings)
    dist_matrix /= np.max(dist_matrix)
    return


def knn_stability(gt_dist_matrix, new_embeddings, k=1000):
    dist_matrix = compute_dist_matrix(new_embeddings)
    dist_matrix /= np.max(dist_matrix)
    dist_matrix = squareform(dist_matrix)

    gt_knn = np.argpartition(gt_dist_matrix, k, axis=0)[:, :k]
    knn = np.argpartition(dist_matrix, k, axis=0)[:, :k]

    intersection_cnt = []
    # import pdb
    # pdb.set_trace()
    for i in range(gt_knn.shape[0]):
        intersection_cnt.append(len(np.intersect1d(gt_knn[i], knn[i])))

    return np.asarray(intersection_cnt)


def clustering_stability(gt_labels, new_embeddings):
    clusters = DBSCAN(min_samples=100).fit(new_embeddings)
    return rand_score(gt_labels, clusters.labels_)


def get_distance_histograms(method):
    files = os.listdir(f'data/{method}')
    for filename in files:
        print(f'At file: {filename}')
        data = pickle.load(open(f'data/{method}/{filename}', 'rb'))
        dist_matrix = compute_dist_matrix(data)
        dist_matrix /= np.max(dist_matrix)
        bins = np.geomspace(1e-10, 1, num=10)
        hist, bin_edges = np.histogram(dist_matrix, bins=bins, density=False)
        plot_data = {
            'hist': hist,
            'bins': bin_edges,
        }
        if not os.path.exists(f'data/{method}_plots'):
            os.makedirs(f'data/{method}_plots')
        pickle.dump(plot_data,
                    open(f'data/{method}_plots/{os.path.splitext(filename)[0]}.pkl', 'wb'))


def get_knn_histogram(method):
    files = os.listdir(f'data/{method}')
    gt_data = pickle.load(open(f'data/pca/pca_50_arpack.pkl', 'rb'))
    gt_dist_matrix = compute_dist_matrix(gt_data)
    gt_dist_matrix /= np.max(gt_dist_matrix)
    gt_dist_matrix = squareform(gt_dist_matrix)
    for filename in files:
        print(f'At file: {filename}')
        if filename.find('pca_50') != -1: # not doing on GT file
            continue
        data = pickle.load(open(f'data/{method}/{filename}', 'rb'))
        knn_plot = knn_stability(gt_dist_matrix, data)
        if not os.path.exists(f'data/{method}_knn_plots'):
            os.makedirs(f'data/{method}_knn_plots')
        pickle.dump(knn_plot,
                    open(f'data/{method}_knn_plots/{os.path.splitext(filename)[0]}.pkl', 'wb'))


def get_cluster_performance(method):
    files = os.listdir(f'data/{method}')
    gt_data = pickle.load(open(f'data/pca/pca_50_arpack.pkl', 'rb'))
    gt_clusters = DBSCAN(min_samples=100).fit(gt_data)
    cluster_scores = dict()
    for filename in files:
        print(f'At file: {filename}')
        if filename.find('pca_50') != -1: # not doing on GT file
            continue
        data = pickle.load(open(f'data/{method}/{filename}', 'rb'))
        rand_score = clustering_stability(gt_clusters.labels_, data)
        print(f'Clustering score is {rand_score}')
        cluster_scores[filename] = rand_score
    if not os.path.exists(f'data/{method}_cluster'):
        os.makedirs(f'data/{method}_cluster')
    pickle.dump(cluster_scores, open(f'data/{method}_cluster/cluster_scores.pkl', 'wb'))



get_distance_histograms('pca')

# TSNE embeddings computation had some issue due to which we had nans in most of them.
# get_distance_histograms('tsne')
get_distance_histograms('umap')

get_knn_histogram('pca')
get_knn_histogram('umap')

get_cluster_performance('pca')
get_cluster_performance('umap')


