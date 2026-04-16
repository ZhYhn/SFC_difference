import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(root_dir, "utils"))

from create_dataset import ConnDataset
from train_MLP import train_mlp
from train_pGCN import train_pgcn
from train_sGCN import train_sgcn
from tqdm import tqdm

# ====== MLP ======

print("MLP...")

train_mlp(os.path.join(root_dir, "Models\\MLP\\mlp.pth"))
train_mlp(os.path.join(root_dir, "Models\\MLP\\mlp_reverse.pth"), reverse=True)
for i in tqdm(range(360), total=360):
    train_mlp(os.path.join(root_dir, f"Models\\direct\\MLP\\mlp_direct_{i}.pth"), direct=i)
    train_mlp(os.path.join(root_dir, f"Models\\direct\\MLP\\mlp_direct_{i}_reverse.pth"), direct=i, reverse=True)

print()



# ====== pGCN & sGCN ======
processed_sc = os.path.join(root_dir, "Data\\train\\sc_dataset\\processed")
processed_fc = os.path.join(root_dir, "Data\\train\\fc_dataset\\processed")
if os.path.isdir(processed_sc):
    os.system(f"rd /s /q {processed_sc}")
if os.path.isdir(processed_fc):
    os.system(f"rd /s /q {processed_fc}")

print("pGCN&sGCN...")

dataset_sc = ConnDataset(conn_type='sc', direct=None)
dataset_fc = ConnDataset(conn_type='fc', direct=None)
train_pgcn(os.path.join(root_dir, "Models\\pGCN\\pgcn.pth"), dataset_sc, dataset_fc)
train_pgcn(os.path.join(root_dir, "Models\\pGCN\\pgcn_reverse.pth"), dataset_fc, dataset_sc) # reverse
train_sgcn(os.path.join(root_dir, "Models\\sGCN\\sgcn_sc.pth"), 
           os.path.join(root_dir, "Models\\sGCN\\sgcn_fc.pth"), dataset_sc, dataset_fc)
os.system(f"rd /s /q {processed_sc}")
os.system(f"rd /s /q {processed_fc}")

print()

for i in range(360):
    if os.path.isdir(processed_sc):
        os.system(f"rd /s /q {processed_sc}")
    if os.path.isdir(processed_fc):
        os.system(f"rd /s /q {processed_fc}")

    print(f'{i+1} / 360')

    dataset_sc = ConnDataset(conn_type='sc', direct=None)
    dataset_fc = ConnDataset(conn_type='fc', direct=i)
    train_sgcn(os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{i}_FC_sc.pth"),
               os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{i}_FC_fc.pth"), dataset_sc, dataset_fc)
    
    os.system(f"rd /s /q {processed_sc}")
    dataset_sc = ConnDataset(conn_type='sc', direct=i)
    train_pgcn(os.path.join(root_dir, f"Models\\direct\\pGCN\\pgcn_direct_{i}.pth"), dataset_sc, dataset_fc)
    train_pgcn(os.path.join(root_dir, f"Models\\direct\\pGCN\\pgcn_direct_{i}_reverse.pth"), dataset_fc, dataset_sc) # reverse
    train_sgcn(os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{i}_SCFC_sc.pth"),
               os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{i}_SCFC_fc.pth"), dataset_sc, dataset_fc)    
    
    os.system(f"rd /s /q {processed_fc}")
    dataset_fc = ConnDataset(conn_type='fc', direct=None)
    train_sgcn(os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{i}_SC_sc.pth"),
               os.path.join(root_dir, f"Models\\direct\\sGCN\\sgcn_direct_{i}_SC_fc.pth"), dataset_sc, dataset_fc)
    
    print()