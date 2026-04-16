import numpy as np
import torch
import random
from tqdm import tqdm
from torch import nn



def predict_mlp(weights, sc, single_node=None):
    '''
    Input arrays must be 2D.
    '''
    # ====== params for Dropout ======
    dropout_rate = 0.5

    # ====== params for MLP ======
    mlp_dims = [359, 128, 128, 128, 128, 359] if single_node != None else \
    [int(0.5*360*(360-1)), 1024*2, 1024*2, 1024*2, 1024*2, int(0.5*360*(360-1))]

    seed_everything()

    model = MLP(mlp_dims, dropout_rate)
    model.load_state_dict(torch.load(weights))

    sc = sc.astype(np.float32).copy()
    sc = torch.tensor(sc, dtype=torch.float32).detach()

    return get_pFC(model, sc, single_node)



def get_pFC(model, sc, single_node):

    triu_indices = torch.triu_indices(360, 360, offset=1)
    if single_node != None:
        mask = torch.ones(sc.shape[0], dtype=torch.bool)
        mask[single_node] = False

    sc = sc[single_node][mask] if single_node != None else sc[triu_indices[0], triu_indices[1]]
    sc = sc.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        output = model(sc)

        if single_node != None:
            pfc = torch.ones(360, dtype=torch.float32)
            pfc[mask] = output
        else:
            pfc = torch.zeros((360, 360), dtype=torch.float32)
            pfc[triu_indices[0], triu_indices[1]] = output
            pfc = pfc + pfc.t()
            pfc[torch.eye(360, dtype=bool)] = 1.0

        return pfc.numpy()



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MLP(torch.nn.Module):

    def __init__(self, dims, dropout_rate):
        super().__init__()

        in_dim, hidden_dim, out_dim = dims[0], dims[1:-1], dims[-1]

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim, hidden_dim[0]))

        for i in range(len(hidden_dim) - 1):

            self.layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))
            self.layers.append(nn.Linear(hidden_dim[i+1], hidden_dim[i+1]))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(p=dropout_rate))

            self.layers.append(nn.BatchNorm1d(hidden_dim[i+1]))

        self.layers.append(nn.Linear(hidden_dim[-1], out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x