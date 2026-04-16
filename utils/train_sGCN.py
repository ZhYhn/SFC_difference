
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
import math
from model_sGCN import Encoder_sc, Encoder_fc, train, test
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




def train_sgcn(save_path_sc, save_path_fc, dataset_sc, dataset_fc):

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
    # ====== coefs of loss ======
    coefs = [1, 10, 15] # contrastive_loss, d_loss_sc, d_loss_fc
    # ====== params for Adam ======
    sc_lr, fc_lr = 0.01, 0.01
    sc_weight_decay, fc_weight_decay = 1e-3, 1e-3
    # ====== params for Training ======
    batch_size = 16
    num_epochs = 1


    model_sc = Encoder_sc(sc_dims, sc_normalize, add_self_loops, use_batch_norm, dropout_rate).to('cuda')
    model_fc = Encoder_fc(fc_dims, fc_normalize, add_self_loops, use_batch_norm, dropout_rate).to('cuda')
    model_sc.apply(init_weights)
    model_fc.apply(init_weights)
    indices = torch.randperm(len(dataset_sc))
    sc_train_dataloader = DataLoader(dataset_sc, batch_size=batch_size, sampler=indices)
    fc_train_dataloader = DataLoader(dataset_fc, batch_size=batch_size, sampler=indices)
    optimizer_sc = torch.optim.Adam(model_sc.parameters(), lr=sc_lr, weight_decay=sc_weight_decay)
    optimizer_fc = torch.optim.Adam(model_fc.parameters(), lr=fc_lr, weight_decay=fc_weight_decay)

    for epoch in range(num_epochs):
        
        # print()
        # head = f'Epoch {epoch+1}/{num_epochs}:'
        # print(head)
        # print()

        train_nums = len(dataset_sc)
        for idx, (batch_sc, batch_fc) in enumerate(zip(sc_train_dataloader, fc_train_dataloader)):

            assert [sc_id[:6] for sc_id in batch_sc.sample_id] == [fc_id[:6] for fc_id in batch_fc.sample_id]
            
            loss, contrastive_loss, d_loss_sc, d_loss_fc = train(model_sc, model_fc, optimizer_sc, optimizer_fc, batch_sc, batch_fc, coefs)
            # print_train_loss = f'{idx+1}/{math.ceil(train_nums/batch_size)}  loss: {loss:.4f}, contrastive loss: {contrastive_loss:.4f}, d loss sc: {d_loss_sc:.4f}, d loss fc: {d_loss_fc:.4f}'
            # print(print_train_loss)
            # print()

        indices = torch.randperm(len(dataset_sc))
        sc_train_dataloader = DataLoader(dataset_sc, batch_size=batch_size, sampler=indices)
        fc_train_dataloader = DataLoader(dataset_fc, batch_size=batch_size, sampler=indices)


    model_sc.to('cpu')
    model_fc.to('cpu')

    torch.save(model_sc.state_dict(), save_path_sc)
    torch.save(model_fc.state_dict(), save_path_fc)