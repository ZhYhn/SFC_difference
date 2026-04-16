from torch_geometric.nn import VGAE
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import Subset
from tqdm import tqdm
import math
from model_pGCN import Encoder_sc, train, test
from torch import nn
import numpy as np
import random




def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()






def train_pgcn(save_path, dataset_sc, dataset_fc):

    # ====== params for GCNConv ======
    sc_normalize = True
    add_self_loops = False
    # ====== params for BatchNorm and Dropout ======
    use_batch_norm = True
    dropout_rate = 0.5
    # ====== params for Encoder ======
    sc_dims = [360, 256, 256]
    mlp_dims = [512, 64, 1]
    # ====== params for Adam ======
    sc_lr = 0.001
    sc_weight_decay = 1e-5
    # ====== params for Training ======
    batch_size = 16
    num_epochs = 1

    model_sc = Encoder_sc(sc_dims, mlp_dims, sc_normalize, add_self_loops, use_batch_norm, dropout_rate).to('cuda')
    model_sc.apply(init_weights)
    indices = torch.randperm(len(dataset_sc))
    sc_train_dataloader = DataLoader(dataset_sc, batch_size=batch_size, sampler=indices)
    fc_train_dataloader = DataLoader(dataset_fc, batch_size=batch_size, sampler=indices)
    optimizer_sc = torch.optim.Adam(model_sc.parameters(), lr=sc_lr, weight_decay=sc_weight_decay)

    for epoch in range(num_epochs):
        
        # print()
        # head = f'Epoch {epoch+1}/{num_epochs}:'
        # print(head)
        # print()

        train_nums = len(dataset_sc)
        for idx, (batch_sc, batch_fc) in enumerate(zip(sc_train_dataloader, fc_train_dataloader)):

            assert [sc_id[:6] for sc_id in batch_sc.sample_id] == [fc_id[:6] for fc_id in batch_fc.sample_id]

            loss = train(model_sc, optimizer_sc, batch_sc, batch_fc)
            # print_train_loss = f'{idx+1}/{math.ceil(train_nums/batch_size)}  loss: {loss:.4f}'
            # print(print_train_loss)
            # print()

        indices = torch.randperm(len(dataset_sc))
        sc_train_dataloader = DataLoader(dataset_sc, batch_size=batch_size, sampler=indices)
        fc_train_dataloader = DataLoader(dataset_fc, batch_size=batch_size, sampler=indices)

    model_sc.to('cpu')
    torch.save(model_sc.state_dict(), save_path)