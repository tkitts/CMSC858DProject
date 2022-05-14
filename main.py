import pickle
import scanpy as sc
import time

# Running tSNE experiments
n_pcs = list(range(10, 0, -5)) + [2]
perplexities = range(5, 50, 5)

for n_pc in n_pcs:
    for perp in perplexities:
        print(f'Doing tsne for n_pc: {n_pc}, perp: {perp}')
        data = pickle.load(open('data/matrix.pkl', 'rb'))
        sc.tl.pca(data, n_comps=n_pc, svd_solver='arpack')
        sc.tl.tsne(data, use_rep='X_pca', n_pcs=n_pc, perplexity=perp, n_jobs=45, use_fast_tsne=True)
        pickle.dump(data.obsm['X_tsne'], open(f'data/tsne_{n_pc}_{perp}.pkl', 'wb'))

# Running PCA experiments
n_comps = range(2, 50, 2)
svd_solvers = ['arpack']

for n_comp in n_comps:
    for svd_solver in svd_solvers:
        print(f'Doing pca for n_comp: {n_comp}, solver: {svd_solver}')
        data = pickle.load(open('data/matrix.pkl', 'rb'))
        sc.tl.pca(data, n_comps=n_comp, svd_solver=svd_solver)
        pickle.dump(data.obsm['X_pca'], open(f'data/pca_{n_comp}_{svd_solver}.pkl', 'wb'))

# Running the UMAP experiments
n_comps = range(50, 1, -2)
n_neighbours = range(20, 101, 20)
# dists = ['euclidean', 'l1', 'l2', 'manhattan']
dists = ['euclidean']
min_dists = range(1, 5, 1)
min_dists = list(map(lambda x: x / 10, min_dists))

for n_comp in n_comps:
    for n_neighbour in n_neighbours:
        for dist in dists:
            for min_dist in min_dists:
                print(f'Doing umap for n_comp: {n_comp}, neighbour: {n_neighbour}, method: {dist}')
                data = pickle.load(open('data/matrix.pkl', 'rb'))
                sc.tl.pca(data, n_comps=n_comp, svd_solver='arpack')
                sc.pp.neighbors(data, n_neighbors=n_neighbour, n_pcs=n_comp, metric=dist)
                sc.tl.umap(data, n_components=n_comp, min_dist=min_dist)
                pickle.dump(data.obsm['X_umap'], open(f'data/umap_{n_comp}_{n_neighbour}_{dist}_{min_dist}.pkl', 'wb'))

