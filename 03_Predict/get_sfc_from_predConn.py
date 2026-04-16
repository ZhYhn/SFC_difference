import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

import numpy as np
from sklearn.metrics import r2_score
import pickle


def row_pearson_correlation(x, y, dim=1):
    x_centered = x - x.mean(axis=dim, keepdims=True)
    y_centered = y - y.mean(axis=dim, keepdims=True)
    covariance = (x_centered * y_centered).sum(axis=dim)
    x_std = np.linalg.norm(x_centered, axis=dim)
    y_std = np.linalg.norm(y_centered, axis=dim)
    corr = covariance / (x_std * y_std + 1e-8)
    return corr

def sparsify(matrix, sparsification):
    matrix = matrix.copy()
    mask = np.eye(matrix.shape[0], dtype=bool)
    threshold = np.quantile(np.abs(matrix[~mask]), 1-sparsification)
    matrix[np.abs(matrix) < threshold] = 0
    return matrix

def get_sfc_corr(sc, fc):
    return row_pearson_correlation(sc, fc, dim=2)

def get_sfc_regr(pFC, fc):
    sfc = np.zeros((pFC.shape[0], pFC.shape[1]), dtype=np.float32)
    for i in range(fc.shape[0]):
        pFC_i, fc_i = pFC[i], fc[i]
        pFC_i = pFC_i[~np.eye(pFC_i.shape[0], dtype=bool)].reshape(pFC_i.shape[0], pFC_i.shape[0]-1)
        fc_i = fc_i[~np.eye(fc_i.shape[0], dtype=bool)].reshape(fc_i.shape[0], fc_i.shape[0]-1)
        r2 = np.zeros(pFC_i.shape[0], dtype=np.float32)
        for j in range(pFC_i.shape[0]):
            r2[j] = r2_score(fc_i[j, :], pFC_i[j, :])
        n, p = 359, 3
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        sfc[i] = adjusted_r2
    return sfc

def get_sfc_mlp(pFC, fc):
    return row_pearson_correlation(pFC, fc, dim=2)

def get_sfc_pgcn(pFC, fc):
    return row_pearson_correlation(pFC, fc, dim=2)



sc_matrices = []
for root, dirs, files in os.walk(os.path.join(root_dir, r"Data\test\structural_connectivity")):
    for file in files:
        path_sc = os.path.join(root, file)
        sc_matrix = np.load(path_sc)
        for _ in range(4):
            sc_matrices.append(sparsify(sc_matrix, sparsification=0.5))
sc_matrices = np.array(sc_matrices, dtype=np.float32)

fc_matrices = []
for root, dirs, files in os.walk(os.path.join(root_dir, r"Data\test\functional_connectivity")):
    for file in files:
        path_fc = os.path.join(root, file)
        fc_matrix = np.load(path_fc)
        fc_matrices.append(sparsify(fc_matrix, sparsification=0.5))
fc_matrices = np.array(fc_matrices, dtype=np.float32)


pFC_full_regr = np.load(os.path.join(root_dir, r"Data\predConn\Regr\pFC_full.npy"))
pFC_full_mlp = np.load(os.path.join(root_dir, r"Data\predConn\MLP\pFC_full.npy"))
pFC_full_pgcn = np.load(os.path.join(root_dir, r"Data\predConn\pGCN\pFC_full.npy"))

pFC_direct_regr = np.load(os.path.join(root_dir, r"Data\predConn\Regr\pFC_direct.npy"))
pFC_direct_mlp = np.load(os.path.join(root_dir, r"Data\predConn\MLP\pFC_direct.npy"))
pFC_direct_pgcn = np.load(os.path.join(root_dir, r"Data\predConn\pGCN\pFC_direct.npy"))

pSC_full_regr = np.load(os.path.join(root_dir, r"Data\predConn\Regr\pSC_full.npy"))
pSC_full_mlp = np.load(os.path.join(root_dir, r"Data\predConn\MLP\pSC_full.npy"))
pSC_full_pgcn = np.load(os.path.join(root_dir, r"Data\predConn\pGCN\pSC_full.npy"))

pSC_direct_regr = np.load(os.path.join(root_dir, r"Data\predConn\Regr\pSC_direct.npy"))
pSC_direct_mlp = np.load(os.path.join(root_dir, r"Data\predConn\MLP\pSC_direct.npy"))
pSC_direct_pgcn = np.load(os.path.join(root_dir, r"Data\predConn\pGCN\pSC_direct.npy"))


sfc_corr = get_sfc_corr(sc_matrices, fc_matrices)

sfc_regr_sc2fc_full = get_sfc_regr(pFC_full_regr, fc_matrices)
sfc_mlp_sc2fc_full = get_sfc_mlp(pFC_full_mlp, fc_matrices)
sfc_pgcn_sc2fc_full = get_sfc_pgcn(pFC_full_pgcn, fc_matrices)

sfc_regr_sc2fc_direct = get_sfc_regr(pFC_direct_regr, fc_matrices)
sfc_mlp_sc2fc_direct = get_sfc_mlp(pFC_direct_mlp, fc_matrices)
sfc_pgcn_sc2fc_direct = get_sfc_pgcn(pFC_direct_pgcn, fc_matrices)

sfc_regr_fc2sc_full = get_sfc_regr(pSC_full_regr, sc_matrices)
sfc_mlp_fc2sc_full = get_sfc_mlp(pSC_full_mlp, sc_matrices)
sfc_pgcn_fc2sc_full = get_sfc_pgcn(pSC_full_pgcn, sc_matrices)

sfc_regr_fc2sc_direct = get_sfc_regr(pSC_direct_regr, sc_matrices)
sfc_mlp_fc2sc_direct = get_sfc_mlp(pSC_direct_mlp, sc_matrices)
sfc_pgcn_fc2sc_direct = get_sfc_pgcn(pSC_direct_pgcn, sc_matrices)


np.save(os.path.join(root_dir, r"Data\sfc\Corr\sfc_corr.npy"), sfc_corr)
np.save(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_sc2fc_full.npy"), sfc_regr_sc2fc_full)
np.save(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_sc2fc_full.npy"), sfc_mlp_sc2fc_full)
np.save(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_sc2fc_full.npy"), sfc_pgcn_sc2fc_full)
np.save(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_fc2sc_full.npy"), sfc_regr_fc2sc_full)
np.save(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_fc2sc_full.npy"), sfc_mlp_fc2sc_full)
np.save(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_fc2sc_full.npy"), sfc_pgcn_fc2sc_full)
np.save(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_sc2fc_direct.npy"), sfc_regr_sc2fc_direct)
np.save(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_sc2fc_direct.npy"), sfc_mlp_sc2fc_direct)
np.save(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_sc2fc_direct.npy"), sfc_pgcn_sc2fc_direct)
np.save(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_fc2sc_direct.npy"), sfc_regr_fc2sc_direct)
np.save(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_fc2sc_direct.npy"), sfc_mlp_fc2sc_direct)
np.save(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_fc2sc_direct.npy"), sfc_pgcn_fc2sc_direct)



with open(os.path.join(root_dir, r"Data\predConn\sGCN\full.pkl"), "rb") as f:
    full = pickle.load(f)
with open(os.path.join(root_dir, r"Data\predConn\sGCN\pFC_full.pkl"), "rb") as f:
    pFC_full = pickle.load(f)
with open(os.path.join(root_dir, r"Data\predConn\sGCN\pSC_full.pkl"), "rb") as f:
    pSC_full = pickle.load(f)
with open(os.path.join(root_dir, r"Data\predConn\sGCN\direct.pkl"), "rb") as f:
    direct = pickle.load(f)

sfc_sgcn_full = np.array([i[2] for i in full], dtype=np.float32)
sfc_sgcn_pFC_full = np.array([i[2] for i in pFC_full], dtype=np.float32)
sfc_sgcn_pSC_full = np.array([i[2] for i in pSC_full], dtype=np.float32)
sfc_sgcn_direct = np.array([i[2] for i in direct], dtype=np.float32)

np.save(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_full.npy"), sfc_sgcn_full)
np.save(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_pFC_full.npy"), sfc_sgcn_pFC_full)
np.save(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_pSC_full.npy"), sfc_sgcn_pSC_full)
np.save(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_direct.npy"), sfc_sgcn_direct)