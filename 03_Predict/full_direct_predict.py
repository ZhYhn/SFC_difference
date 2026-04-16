import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(root_dir, "utils"))

from predict_Regr import predict_regr
from predict_MLP import predict_mlp
from predict_pGCN import predict_pgcn
from predict_sGCN import predict_sgcn
from tqdm import tqdm
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
        sc_matrix = sparsify(np.load(path_sc), sparsification=0.5)
        for _ in range(4):
            sc_matrices.append(sc_matrix)
sc_matrices = np.array(sc_matrices).astype(np.float32)

fc_matrices = []
for root, dirs, files in os.walk(os.path.join(root_dir, "Data\\test\\functional_connectivity")):
    for file in files:
        path_fc = os.path.join(root, file)
        fc_matrix = sparsify(np.load(path_fc), sparsification=0.5)
        fc_matrices.append(fc_matrix)
fc_matrices = np.array(fc_matrices).astype(np.float32)


def get_pred_regr(a, b):
    pred_full = predict_regr(a, b, single_node=None)
    pred_direct = np.zeros_like(pred_full)
    for node_i in range(a.shape[0]):
        a_direct = np.zeros_like(a)
        a_direct[node_i, :] = a[node_i, :]
        a_direct[:, node_i] = a[:, node_i]
        pred_direct[node_i] = predict_regr(a_direct, b, single_node=node_i)
    return pred_full, pred_direct


def get_pred_mlp(a, reverse=False):
    model = os.path.join(root_dir, "Models\\MLP\\mlp.pth") \
        if not reverse else os.path.join(root_dir, "Models\\MLP\\mlp_reverse.pth")
    pred_full = predict_mlp(model, a, single_node=None)
    pred_direct = np.zeros_like(pred_full)
    for node_i in range(a.shape[0]):
        model_i = os.path.join(root_dir, f"Models\\direct\\MLP\\mlp_direct_{node_i}.pth") \
            if not reverse else os.path.join(root_dir, f"Models\\direct\\MLP\\mlp_direct_{node_i}_reverse.pth")
        a_direct = np.zeros_like(a)
        a_direct[node_i, :] = a[node_i, :]
        a_direct[:, node_i] = a[:, node_i]
        pred_direct[node_i] = predict_mlp(model_i, a_direct, single_node=node_i)
    return pred_full, pred_direct


def get_pred_pgcn(a, b, reverse=False):
    model = os.path.join(root_dir, "Models\\pGCN\\pgcn.pth") \
        if not reverse else os.path.join(root_dir, "Models\\pGCN\\pgcn_reverse.pth")
    pred_full = predict_pgcn(model, a, b, single_node=None)
    pred_direct = np.zeros_like(pred_full)
    for node_i in range(a.shape[0]):
        model_i = os.path.join(root_dir, f"Models\\direct\\pGCN\\pgcn_direct_{node_i}.pth") \
            if not reverse else os.path.join(root_dir, f"Models\\direct\\pGCN\\pgcn_direct_{node_i}_reverse.pth")
        a_direct = np.zeros_like(a)
        a_direct[node_i, :] = a[node_i, :]
        a_direct[:, node_i] = a[:, node_i]
        pred_direct[node_i] = predict_pgcn(model_i, a_direct, b, single_node=node_i)
    return pred_full, pred_direct


def get_pred_sgcn(sc, fc):
    model_full = (os.path.join(root_dir, "Models\\sGCN\\sgcn_sc.pth"), 
                  os.path.join(root_dir, "Models\\sGCN\\sgcn_fc.pth"))
    
    full = predict_sgcn(model_full[0], model_full[1], sc, fc)
    pFC_full = (np.zeros_like(sc), np.zeros_like(fc), np.zeros(sc.shape[0]))
    pSC_full = (np.zeros_like(sc), np.zeros_like(fc), np.zeros(sc.shape[0]))
    direct = (np.zeros_like(sc), np.zeros_like(fc), np.zeros(sc.shape[0]))

    for node_i in range(sc.shape[0]):

        model_i_direct_FC = (os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{node_i}_FC_sc.pth"), 
                             os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{node_i}_FC_fc.pth"))
        model_i_direct_SC = (os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{node_i}_SC_sc.pth"), 
                             os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{node_i}_SC_fc.pth"))
        model_i_direct = (os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{node_i}_SCFC_sc.pth"), 
                          os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{node_i}_SCFC_fc.pth"))
        
        sc_direct = np.zeros_like(sc)
        sc_direct[node_i, :] = sc[node_i, :]
        sc_direct[:, node_i] = sc[:, node_i]
        fc_direct = np.zeros_like(fc)
        fc_direct[node_i, :] = fc[node_i, :]
        fc_direct[:, node_i] = fc[:, node_i]

        for i in range(3):
            pFC_full[i][node_i] = predict_sgcn(model_i_direct_FC[0], model_i_direct_FC[1], sc, fc_direct)[i][node_i]
            pSC_full[i][node_i] = predict_sgcn(model_i_direct_SC[0], model_i_direct_SC[1], sc_direct, fc)[i][node_i]
            direct[i][node_i] = predict_sgcn(model_i_direct[0], model_i_direct[1], sc_direct, fc_direct)[i][node_i]

    return full, pFC_full, pSC_full, direct



pFC_full_regr = np.zeros_like(sc_matrices)
pFC_direct_regr = np.zeros_like(sc_matrices)
pSC_full_regr = np.zeros_like(sc_matrices)
pSC_direct_regr = np.zeros_like(sc_matrices)

pFC_full_mlp = np.zeros_like(sc_matrices)
pFC_direct_mlp = np.zeros_like(sc_matrices)
pSC_full_mlp = np.zeros_like(sc_matrices)
pSC_direct_mlp = np.zeros_like(sc_matrices)

pFC_full_pgcn = np.zeros_like(sc_matrices)
pFC_direct_pgcn = np.zeros_like(sc_matrices)
pSC_full_pgcn = np.zeros_like(sc_matrices)
pSC_direct_pgcn = np.zeros_like(sc_matrices)

full_sgcn = []
pFC_full_sgcn = []
pSC_full_sgcn = []
direct_sgcn = []


for subj in tqdm(range(sc_matrices.shape[0]), total=sc_matrices.shape[0]):
    sc, fc = sc_matrices[subj], fc_matrices[subj]
    
    # ====== Regr ======
    pFC_full, pFC_direct = get_pred_regr(sc, fc)
    pFC_full_regr[subj] = pFC_full
    pFC_direct_regr[subj] = pFC_direct
    pSC_full, pSC_direct = get_pred_regr(fc, sc) # reverse
    pSC_full_regr[subj] = pSC_full
    pSC_direct_regr[subj] = pSC_direct

    # ====== MLP ======
    pFC_full, pFC_direct = get_pred_mlp(sc, reverse=False)
    pFC_full_mlp[subj] = pFC_full
    pFC_direct_mlp[subj] = pFC_direct
    pSC_full, pSC_direct = get_pred_mlp(fc, reverse=True) # reverse
    pSC_full_mlp[subj] = pSC_full
    pSC_direct_mlp[subj] = pSC_direct

    # ====== pGCN ======
    pFC_full, pFC_direct = get_pred_pgcn(sc, fc, reverse=False)
    pFC_full_pgcn[subj] = pFC_full
    pFC_direct_pgcn[subj] = pFC_direct
    pSC_full, pSC_direct = get_pred_pgcn(fc, sc, reverse=True) # reverse
    pSC_full_pgcn[subj] = pSC_full
    pSC_direct_pgcn[subj] = pSC_direct

    # ====== sGCN ======
    full, pFC_full, pSC_full, direct = get_pred_sgcn(sc, fc)
    full_sgcn.append(full)
    pFC_full_sgcn.append(pFC_full)
    pSC_full_sgcn.append(pSC_full)
    direct_sgcn.append(direct)


    np.save(os.path.join(root_dir, "Data\\predConn\\Regr\\pFC_full.npy"), pFC_full_regr)
    np.save(os.path.join(root_dir, "Data\\predConn\\Regr\\pFC_direct.npy"), pFC_direct_regr)
    np.save(os.path.join(root_dir, "Data\\predConn\\Regr\\pSC_full.npy"), pSC_full_regr)
    np.save(os.path.join(root_dir, "Data\\predConn\\Regr\\pSC_direct.npy"), pSC_direct_regr)

    np.save(os.path.join(root_dir, "Data\\predConn\\MLP\\pFC_full.npy"), pFC_full_mlp)
    np.save(os.path.join(root_dir, "Data\\predConn\\MLP\\pFC_direct.npy"), pFC_direct_mlp)
    np.save(os.path.join(root_dir, "Data\\predConn\\MLP\\pSC_full.npy"), pSC_full_mlp)
    np.save(os.path.join(root_dir, "Data\\predConn\\MLP\\pSC_direct.npy"), pSC_direct_mlp)

    np.save(os.path.join(root_dir, "Data\\predConn\\pGCN\\pFC_full.npy"), pFC_full_pgcn)
    np.save(os.path.join(root_dir, "Data\\predConn\\pGCN\\pFC_direct.npy"), pFC_direct_pgcn)
    np.save(os.path.join(root_dir, "Data\\predConn\\pGCN\\pSC_full.npy"), pSC_full_pgcn)
    np.save(os.path.join(root_dir, "Data\\predConn\\pGCN\\pSC_direct.npy"), pSC_direct_pgcn)

    with open(os.path.join(root_dir, "Data\\predConn\\sGCN\\full.pkl"), "wb") as f:
        pickle.dump(full_sgcn, f)
    with open(os.path.join(root_dir, "Data\\predConn\\sGCN\\pFC_full.pkl"), "wb") as f:
        pickle.dump(pFC_full_sgcn, f)
    with open(os.path.join(root_dir, "Data\\predConn\\sGCN\\pSC_full.pkl"), "wb") as f:
        pickle.dump(pSC_full_sgcn, f)
    with open(os.path.join(root_dir, "Data\\predConn\\sGCN\\direct.pkl"), "wb") as f:
        pickle.dump(direct_sgcn, f)