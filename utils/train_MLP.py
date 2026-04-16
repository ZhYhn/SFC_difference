from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch import nn
import random
import math
from tqdm import tqdm
import os



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



# ====== thresholds for sparsification ======
sc_p = 0.5
fc_p = 0.5
# ====== params for Dropout ======
dropout_rate = 0.5
# ====== params for Adam ======
lr = 0.001
weight_decay = 1e-5
# ====== params for Training ======
batch_size = 16
num_epochs = 4






class MLPDataset(Dataset):
    
    def __init__(self, sc_paths, fc_paths, direct):
        self.sc_paths = sc_paths
        self.fc_paths = fc_paths
        self.direct = direct

    def __len__(self):
        return len(self.sc_paths)
    
    def __getitem__(self, idx):
        sc_path = self.sc_paths[idx]
        fc_path = self.fc_paths[idx]

        assert sc_path.split('\\')[-1].split('_')[0] == fc_path.split('\\')[-1].split('_')[0]

        sc = np.load(sc_path)
        fc = np.load(fc_path)

        mask = np.eye(sc.shape[0], dtype=bool)
        sc_threshold = np.quantile(sc[~mask], 1 - sc_p)
        fc_threshold = np.quantile(np.abs(fc[~mask]), 1 - fc_p)
        sc[sc < sc_threshold] = 0
        fc[np.abs(fc) < fc_threshold] = 0

        if self.direct == None:
            sc = sc[np.triu_indices_from(sc, k=1)]
            fc = fc[np.triu_indices_from(fc, k=1)]
        else:
            sc = np.delete(sc[self.direct], self.direct)
            fc = np.delete(fc[self.direct], self.direct)

        return torch.tensor(sc, dtype=torch.float32), torch.tensor(fc, dtype=torch.float32)




class MLP(torch.nn.Module):

    def __init__(self, dims):
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






def train_mlp(save_path, direct=None, reverse=False):

    seed_everything()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    if direct == None:
        mlp_dims = [int(0.5*360*(360-1)), 1024*2, 1024*2, 1024*2, 1024*2, int(0.5*360*(360-1))]
    else:
        mlp_dims = [359, 128, 128, 128, 128, 359]


    sc_paths = []
    fc_paths = []
    for root, _, files in os.walk(os.path.join(root_dir, 'Data\\train\\sc_dataset', 'raw')):
        for file in files:
            if file.endswith('.npy'):
                for _ in range(4):
                    sc_paths.append(os.path.join(root, file))
    for root, _, files in os.walk(os.path.join(root_dir, 'Data\\train\\fc_dataset', 'raw')):
        for file in files:
            if file.endswith('.npy'):
                fc_paths.append(os.path.join(root, file))


    model = MLP(mlp_dims).to('cuda')
    model.apply(init_weights)
    train_dataset = MLPDataset(sc_paths, fc_paths, direct) if not reverse else MLPDataset(fc_paths, sc_paths, direct)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    for epoch in range(num_epochs):
        # print()
        # print(f'Epoch {epoch+1}/{num_epochs}:')
        # print()
        model.train()
        train_nums = len(train_dataset)
        for idx, (sc_batch, fc_batch) in enumerate(train_dataloader):
            sc_batch = sc_batch.to('cuda')
            fc_batch = fc_batch.to('cuda')
            optimizer.zero_grad()
            output = model(sc_batch)
            loss = nn.MSELoss()(output, fc_batch)
            loss.backward()
            optimizer.step()
            # print(f'{idx+1}/{math.ceil(train_nums/batch_size)}  loss: {loss:.4f}')
            # print()


    model.to('cpu')
    torch.save(model.state_dict(), save_path)