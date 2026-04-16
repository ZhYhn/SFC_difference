import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

import pickle
import numpy as np



def sparsify(matrix, sparsification):
    matrix = matrix.copy()
    mask = np.eye(matrix.shape[0], dtype=bool)
    threshold = np.quantile(np.abs(matrix[~mask]), 1-sparsification)
    matrix[np.abs(matrix) < threshold] = 0
    return matrix

sc_matrices = []
for root, dirs, files in os.walk(os.path.join(root_dir, "Data\\test\\structural_connectivity")):
    for file in files:
        path_sc = os.path.join(root, file)
        sc_matrix = np.load(path_sc)
        for _ in range(4):
            sc_matrices.append(sparsify(sc_matrix, sparsification=0.5))
sc_matrices = np.array(sc_matrices).astype(np.float32)

fc_matrices = []
for root, dirs, files in os.walk(os.path.join(root_dir, "Data\\test\\functional_connectivity")):
    for file in files:
        path_fc = os.path.join(root, file)
        fc_matrix = np.load(path_fc)
        fc_matrices.append(sparsify(fc_matrix, sparsification=0.5))
fc_matrices = np.array(fc_matrices).astype(np.float32)


def symmetrize(matrices):
    if len(matrices.shape) == 2:
        matrices = (matrices + matrices.T) / 2
        np.fill_diagonal(matrices, 0)
    elif len(matrices.shape) == 3:
        matrices = (matrices + np.transpose(matrices, (0, 2, 1))) / 2
        for i in range(matrices.shape[0]):
            np.fill_diagonal(matrices[i], 0)
    return matrices



for approach in ["Regr", "MLP", "pGCN"]:
    pFC_full = np.load(os.path.join(root_dir, f"Data\\predConn\\{approach}\\pFC_full.npy"))
    pFC_direct = np.load(os.path.join(root_dir, f"Data\\predConn\\{approach}\\pFC_direct.npy"))
    pSC_full = np.load(os.path.join(root_dir, f"Data\\predConn\\{approach}\\pSC_full.npy"))
    pSC_direct = np.load(os.path.join(root_dir, f"Data\\predConn\\{approach}\\pSC_direct.npy"))

    pFC_full = symmetrize(pFC_full)
    pFC_direct = symmetrize(pFC_direct)
    pSC_full = symmetrize(pSC_full)
    pSC_direct = symmetrize(pSC_direct)

    diff_i_sc = np.abs(pFC_full-fc_matrices) - np.abs(pFC_direct-fc_matrices)
    diff_i_fc = np.abs(pSC_full-sc_matrices) - np.abs(pSC_direct-sc_matrices)
    diff_m_sc = np.abs(pFC_direct - fc_matrices) - np.abs(sc_matrices - fc_matrices)
    diff_m_fc = np.abs(pSC_direct - sc_matrices) - np.abs(fc_matrices - sc_matrices)

    np.save(os.path.join(root_dir, f"Data\\difference\\{approach}\\diff_i_sc.npy"), diff_i_sc)
    np.save(os.path.join(root_dir, f"Data\\difference\\{approach}\\diff_i_fc.npy"), diff_i_fc)
    np.save(os.path.join(root_dir, f"Data\\difference\\{approach}\\diff_m_sc.npy"), diff_m_sc)
    np.save(os.path.join(root_dir, f"Data\\difference\\{approach}\\diff_m_fc.npy"), diff_m_fc)