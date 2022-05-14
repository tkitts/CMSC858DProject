from asyncio import constants
import scanpy as sc
import anndata as ad
from sklearn import metrics
import pickle
from pathlib import Path


# this method generates a cluster for an embed, though it assumes neighbors have been computed
def clustermetric(adata: ad.AnnData, GT: ad.AnnData):
    # use the Leiden graph-cluster method
    adata = sc.tl.leiden(adata, copy=True)
    # sc.pl.embedding(adata, color='leiden', basis='pca')
    return metrics.cluster.adjusted_rand_score(adata.obs.to_numpy().flatten(), GT.obs.to_numpy().flatten())


# this method reads a pickle file and outputs the rand_score for neighbors 5-50 in steps of 5
def readEmbed(inFile: str):
    fileSplit = inFile.split('_')
    dim = int(fileSplit[1])
    # nneighbor = int(fileSplit[2])
    # distmetric = fileSplit[3]
    # min_dist = float(fileSplit[4][0:3])

    output = []
    with open(inFile, 'rb') as f:
        adata = ad.AnnData(pickle.load(f))

    for e in range(10, 51, 5):
        sc.pp.neighbors(adata, n_pcs=dim, n_neighbors=e, metric='euclidean', copy=False)
        with open('GTC_' + str(e), 'rb') as f:
            GT = ad.AnnData(pickle.load(f))
        output.append(clustermetric(adata, GT))

    return output


# this method generates GT clusters for n_neighbors
def genGT():
    GTBase = sc.read_10x_h5('10k_PBMC_3p_nextgem_Chromium_X_raw_feature_bc_matrix.h5')
    for e in range(10, 51, 5):
        GT = GTBase.copy()
        sc.pp.neighbors(GT, n_pcs=GT.X, n_neighbors=e, metric='euclidean', copy=False, use_rep=None)
        sc.tl.leiden(GT, copy=False)
        with open('GTC_' + str(e), 'wb') as f:
            pickle.dump(GT, f)
    return


genGT()
# files = Path('\umap')
# for file in files:
#   readEmbed(file)