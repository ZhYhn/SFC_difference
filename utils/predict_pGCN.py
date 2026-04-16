import numpy as np
from model_pGCN import Encoder_sc
import torch
import random
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from tqdm import tqdm



def predict_pgcn(weights_sc, sc, fc, single_node=None):
    '''
    Input arrays must be 2D.
    '''
    # ====== params for GCNConv ======
    sc_normalize = True
    add_self_loops = False

    # ====== params for BatchNorm and Dropout ======
    use_batch_norm = True
    dropout_rate = 0.5

    # ====== params for Encoder ======
    sc_dims = [360, 256, 256]
    mlp_dims = [512, 64, 1]

    seed_everything()

    model_sc = Encoder_sc(sc_dims, mlp_dims, sc_normalize, add_self_loops, use_batch_norm, dropout_rate)
    model_sc.load_state_dict(torch.load(weights_sc))

    sc, fc = sc.astype(np.float32).copy(), fc.astype(np.float32).copy()

    sc = torch.tensor(sc, dtype=torch.float32).detach()
    fc = torch.tensor(fc, dtype=torch.float32).detach()
    if single_node != None:
        return get_pFC_single_node(model_sc, sc, fc, single_node)
    else:
        return get_pFC(model_sc, sc, fc)



def get_pFC(model, sc, fc):

    x = torch.tensor(np.eye(360), dtype=torch.float)
    edge_index_sc, edge_attr_sc = dense_to_sparse(sc)
    edge_index_fc, _ = dense_to_sparse(fc)

    data_sc = Data(x=x, edge_index=edge_index_sc, edge_attr=edge_attr_sc)

    model.eval()

    with torch.no_grad():
        output = model(data_sc, edge_index_fc).view(-1)

        pFC = torch.zeros((360, 360), dtype=torch.float32)

        for node_i in range(sc.shape[0]):
            pFC[node_i, edge_index_fc[1, edge_index_fc[0]==node_i]] = output[edge_index_fc[0]==node_i]

        return pFC.numpy()



def get_pFC_single_node(model, sc, fc, node_i):

    x = torch.tensor(np.eye(360), dtype=torch.float)
    edge_index_sc, edge_attr_sc = dense_to_sparse(sc)
    edge_index_fc, _ = dense_to_sparse(fc)

    data_sc = Data(x=x, edge_index=edge_index_sc, edge_attr=edge_attr_sc)

    model.eval()

    with torch.no_grad():
        output = model(data_sc, edge_index_fc).view(-1)

        pFC = torch.zeros(360, dtype=torch.float32)

        pFC[edge_index_fc[1, edge_index_fc[0]==node_i]] = output[edge_index_fc[0]==node_i]

        return pFC.numpy()



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False