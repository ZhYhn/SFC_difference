import numpy as np
from model_sGCN import Encoder_sc, Encoder_fc
import torch
import random
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from tqdm import tqdm



def predict_sgcn(weights_sc, weights_fc, sc, fc):
    '''
    Input arrays must be 2D.
    '''
    # ====== params for GCNConv ======
    sc_normalize = True
    fc_normalize = True
    add_self_loops = False

    # ====== params for BatchNorm and Dropout ======
    use_batch_norm = True
    dropout_rate = 0.2

    # ====== params for Encoder ======
    sc_dims = [360, 128, 32]
    fc_dims = [360, 128, 32]

    seed_everything()

    model_sc = Encoder_sc(sc_dims, sc_normalize, add_self_loops, use_batch_norm, dropout_rate)
    model_fc = Encoder_fc(fc_dims, fc_normalize, add_self_loops, use_batch_norm, dropout_rate)

    model_sc.load_state_dict(torch.load(weights_sc))
    model_fc.load_state_dict(torch.load(weights_fc))

    sc, fc = sc.copy().astype(np.float32), fc.copy().astype(np.float32)

    sc = torch.tensor(sc, dtype=torch.float32).detach()
    fc = torch.tensor(fc, dtype=torch.float32).detach()
    return get_latentConn(model_sc, model_fc, sc, fc)
                


def get_latentConn(model_sc, model_fc, sc, fc):

    x = torch.tensor(np.eye(360), dtype=torch.float)
    edge_index_sc, edge_attr_sc = dense_to_sparse(sc)
    edge_index_fc, edge_attr_fc = dense_to_sparse(fc)

    data_sc = Data(x=x, edge_index=edge_index_sc, edge_attr=edge_attr_sc)
    data_fc = Data(x=x, edge_index=edge_index_fc, edge_attr=edge_attr_fc)

    model_sc.eval()
    model_fc.eval()

    with torch.no_grad():
        z_sc_nodes = model_sc(data_sc)
        z_fc_nodes = model_fc(data_fc)

        distance_sc = torch.cdist(z_sc_nodes, z_sc_nodes, p=2).numpy().astype(np.float32)
        # sigma = np.median(distance_sc)
        # latentConn_sc = np.exp(-distance_sc**2 / (2 * sigma**2))

        distance_fc = torch.cdist(z_fc_nodes, z_fc_nodes, p=2).numpy().astype(np.float32)
        # sigma = np.median(distance_fc)
        # latentConn_fc = np.exp(-distance_fc**2 / (2 * sigma**2))

        distance = torch.sum((z_sc_nodes - z_fc_nodes) ** 2, dim=1).detach().numpy().astype(np.float32)
        sigma = np.median(np.sqrt(distance))
        sfc = np.exp(-distance / (2 * sigma**2))

        return distance_sc, distance_fc, sfc



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False