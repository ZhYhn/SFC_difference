import numpy as np
import matplotlib.pyplot as plt
import os
import umap
from scipy import stats
from scipy.spatial.distance import cdist
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

corr = np.load(os.path.join(root_dir, r"Data\sfc\Corr\sfc_corr.npy"))
regr = np.load(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_sc2fc_full.npy"))
mlp = np.load(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_sc2fc_full.npy"))
pgcn = np.load(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_sc2fc_full.npy"))
sgcn = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_full.npy"))

boundaries = [
    [0, 30],
    [30, 49],
    [49, 76],
    [76, 88],
    [88, 102],
    [102, 124],
    [124, 132],
    [132, 172],
    [172, 177],
    [177, 180],
    [180, 30+180],
    [30+180, 49+180],
    [49+180, 76+180],
    [76+180, 88+180],
    [88+180, 102+180],
    [102+180, 124+180],
    [124+180, 132+180],
    [132+180, 172+180],
    [172+180, 177+180],
    [177+180, 180+180]
]

def reorganize(sfc, network=True):
    tab = pd.read_excel(os.path.join(root_dir, r"others\allTables.xlsx"), sheet_name="table 2")
    transf = pd.concat([tab.iloc[:, [0, 2]], tab.iloc[:, [4, 6]].rename(columns={'Idx..1':'Idx.', 'Orig..1':'Orig.'}), tab.iloc[:, [8, 10]].rename(columns={'Idx..2':'Idx.', 'Orig..2':'Orig.'})], axis=0)
    transf = transf.sort_values(by='Orig.').reset_index(drop=True)
    sfc_new = np.zeros_like(sfc)
    for i in range(360):
        idx = transf.iloc[i, 0]-1 if i < 180 else transf.iloc[i-180, 0]-1+180
        if sfc.ndim == 2:
            sfc_new[:, idx] = sfc[:, i]
        elif sfc.ndim == 1:
            sfc_new[idx] = sfc[i]
    if not network:
        return sfc_new
    boundaries = [
        [0, 30],
        [30, 49],
        [49, 76],
        [76, 88],
        [88, 102],
        [102, 124],
        [124, 132],
        [132, 172],
        [172, 177],
        [177, 180],
        [180, 30+180],
        [30+180, 49+180],
        [49+180, 76+180],
        [76+180, 88+180],
        [88+180, 102+180],
        [102+180, 124+180],
        [124+180, 132+180],
        [132+180, 172+180],
        [172+180, 177+180],
        [177+180, 180+180]
    ]
    results = np.zeros((400, 20)) if sfc.ndim == 2 else np.zeros(20)
    for idx, boundary in enumerate(boundaries):
        start, end = boundary
        if sfc.ndim == 2:
            results[:, idx] = sfc_new[:, start:end].mean(axis=1)
        elif sfc.ndim == 1:
            results[idx] = sfc_new[start:end].mean()
    return results


def min_max_scaling(sfc):
    if sfc.ndim == 2:
        row_min = sfc.min(axis=1, keepdims=True)
        row_max = sfc.max(axis=1, keepdims=True)
        return (sfc - row_min) / (row_max - row_min)
    elif sfc.ndim == 1:
        return (sfc - sfc.min()) / (sfc.max() - sfc.min())


network_names = ['L Visual','L Somatomotor','L Cingulo-opercular','L Dorsal attention','L Language',
                 'L Frontoparietal','L Auditory','L Default mode','L Multimodal','L Orbito-affective',
                 'R Visual','R Somatomotor','R Cingulo-opercular','R Dorsal attention','R Language',
                 'R Frontoparietal','R Auditory','R Default mode','R Multimodal','R Orbito-affective']
network_to_idx = {'L Visual':1,'L Somatomotor':2,'L Cingulo-opercular':3,'L Dorsal attention':4,'L Language':5,
               'L Frontoparietal':6,'L Auditory':7,'L Default mode':8,'L Multimodal':9,'L Orbito-affective':10,
               'R Visual':11,'R Somatomotor':12,'R Cingulo-opercular':13,'R Dorsal attention':14,'R Language':15,
               'R Frontoparietal':16,'R Auditory':17,'R Default mode':18,'R Multimodal':19,'R Orbito-affective':20}

approach_data = np.stack([reorganize(min_max_scaling(corr)), reorganize(min_max_scaling(regr)), 
                          reorganize(min_max_scaling(mlp)), reorganize(min_max_scaling(pgcn)), 
                          reorganize(min_max_scaling(sgcn))], axis=-1)

network_sfc = approach_data.mean(axis=0).mean(axis=1)
global_mean = network_sfc.mean()
deviation = np.abs(network_sfc - global_mean)
norm = Normalize(vmin=deviation.min(), vmax=deviation.max())
norm_dev = norm(deviation)
cmap = plt.cm.coolwarm
colors = [cmap(d) for d in norm_dev] 

variance_matrix = np.var(approach_data, axis=-1, ddof=1)
df_var = pd.DataFrame(variance_matrix, 
                      columns=network_names)
df_var['Sample'] = np.arange(400)
df_var_melted = df_var.melt(id_vars='Sample', 
                            var_name='Network', 
                            value_name='Variance')

network_var_medians = df_var_melted.groupby('Network')['Variance'].median().sort_values()
ordered_networks = network_var_medians.index.tolist()
ordered_colors = [colors[network_to_idx[net]-1] for net in ordered_networks]

valid_approach_data = np.stack([reorganize(min_max_scaling(corr)), reorganize(min_max_scaling(regr)), 
                                reorganize(min_max_scaling(mlp)), reorganize(min_max_scaling(pgcn))], axis=-1)
variance_4approaches = np.var(valid_approach_data, axis=-1, ddof=1)
median_4approaches = np.median(variance_4approaches, axis=0)
network_to_median4 = {network_names[i]: median_4approaches[i] for i in range(20)}
ordered_median4 = [network_to_median4[net] for net in ordered_networks]

plt.figure(figsize=(16, 6))
ax = sns.boxplot(data=df_var_melted, x='Network', y='Variance', 
                 order=ordered_networks, palette=ordered_colors)

x_positions = range(len(ordered_networks))
for x, y in zip(x_positions, ordered_median4):
    ax.hlines(y=y, xmin=x - 0.4, xmax=x + 0.4,
              colors='black', linestyles='dashed', linewidth=2)

plt.xticks(rotation=45, fontsize=12)
plt.xlabel('Functional network', fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('Variance', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(root_dir, f"Data\\figs\\01_sfc_validation\\networkSFC_boxplot.png"), dpi=600, bbox_inches='tight')



def min_max_scaling(sfc):
    if sfc.ndim == 2:
        row_min = sfc.min(axis=1, keepdims=True)
        row_max = sfc.max(axis=1, keepdims=True)
        return (sfc - row_min) / (row_max - row_min)
    elif sfc.ndim == 1:
        return (sfc - sfc.min()) / (sfc.max() - sfc.min())

regional_approach_data = np.stack([min_max_scaling(reorganize(corr, network=False).mean(axis=0)), min_max_scaling(reorganize(regr, network=False).mean(axis=0)), 
                                   min_max_scaling(reorganize(mlp, network=False).mean(axis=0)), min_max_scaling(reorganize(pgcn, network=False).mean(axis=0)), 
                                   min_max_scaling(reorganize(sgcn, network=False).mean(axis=0))], axis=0)

approach_names = ['Correlation', 'Regression', 'MLP', 'pGCN', 'sGCN']

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
sns.heatmap(regional_approach_data[:, :180], cmap='coolwarm', center=0.5,
            ax=ax[0], cbar=False)
sns.heatmap(regional_approach_data[:, 180:], cmap='coolwarm', center=0.5,
            ax=ax[1], cbar=False)

ax[0].set_xlim(-0.5, 180.5)
ax[0].set_ylim(5.05, -0.05)
ax[1].set_xlim(-0.5, 180.5)
ax[1].set_ylim(5.05, -0.05)

for idx, boundary in enumerate(boundaries[:10]):
    start, end = boundary
    width = end - start
    row = idx % 5
    rect = Rectangle((start, row), width, 1,
                     linewidth=1.5, edgecolor='black', facecolor='none')
    ax[0].add_patch(rect)

for idx, boundary in enumerate(boundaries[10:]):
    global_idx = idx + 10
    start, end = boundary
    start_mapped = start - 180
    end_mapped = end - 180
    width = end_mapped - start_mapped
    row = global_idx % 5
    rect = Rectangle((start_mapped, row), width, 1,
                     linewidth=1.5, edgecolor='black', facecolor='none')
    ax[1].add_patch(rect)

ax[0].set_xticks([])
ax[0].set_xticklabels([])

xticks_right = [(start + end) / 2 - 180 for start, end in boundaries[10:]]
ax[1].set_xticks(xticks_right)
ax[1].set_xticklabels([net[2:] for net in network_names[10:]], rotation=45, fontsize=12)

ax[0].set_yticklabels(approach_names, rotation=0, fontsize=12)
ax[1].set_yticklabels(approach_names, rotation=0, fontsize=12)

fig.text(0.5, 0.02, 'Functional network', ha='center', va='center', fontsize=15)
fig.text(0, 0.55, 'Approach', ha='center', va='center', rotation='vertical', fontsize=15)
ax[0].set_title('Left', y=1.02, size=20)
ax[1].set_title('Right', y=1.02, size=20)

plt.tight_layout(rect=[0, 0, 0.9, 1])
im = ax[0].collections[0]
cbar_ax = fig.add_axes([0.89, 0.3, 0.02, 0.5])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
plt.savefig(os.path.join(root_dir, f"Data\\figs\\01_sfc_validation\\networkSFC_heatplot_new.png"), dpi=600, bbox_inches='tight')



def sparsify(matrix, sparsification):
    matrix = matrix.copy()
    mask = np.eye(matrix.shape[0], dtype=bool)
    threshold = np.quantile(np.abs(matrix[~mask]), 1-sparsification)
    matrix[np.abs(matrix) < threshold] = 0
    return matrix

mlp_full_pFC = np.load(os.path.join(script_dir, r"predConn\MLP\pFC_full.npy")).mean(axis=0)
pgcn_full_pFC = np.load(os.path.join(script_dir, r"predConn\pGCN\pFC_full.npy")).mean(axis=0)

mlp_full_pSC = np.load(os.path.join(script_dir, r"predConn\MLP\pSC_full.npy")).mean(axis=0)
pgcn_full_pSC = np.load(os.path.join(script_dir, r"predConn\pGCN\pSC_full.npy")).mean(axis=0)

sc_matrices = []
for root, dirs, files in os.walk(os.path.join(script_dir, r"test\structural_connectivity")):
    for file in files:
        path_sc = os.path.join(root, file)
        sc_matrix = np.load(path_sc).astype(np.float32)
        for _ in range(4):
            sc_matrices.append(sparsify(sc_matrix, sparsification=0.5))
sc = np.array(sc_matrices).astype(np.float32).mean(axis=0)

fc_matrices = []
for root, dirs, files in os.walk(os.path.join(script_dir, r"test\functional_connectivity")):
    for file in files:
        path_fc = os.path.join(root, file)
        fc_matrix = np.load(path_fc)
        fc_matrices.append(sparsify(fc_matrix, sparsification=0.5))
fc = np.array(fc_matrices).astype(np.float32).mean(axis=0)


def predConn_plot(pfc, efc, title, save_path):
    indices = np.triu_indices(pfc.shape[0], k=1)
    pfc[indices], efc[indices] = 0, 0
    matrix = efc + pfc.T
    matrix[np.eye(pfc.shape[0], dtype=bool)] = 0
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(matrix, 
                    cmap='viridis', 
                    center=0, 
                    square=True,
                    xticklabels=False,
                    yticklabels=False, 
                    cbar=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)
    plt.title(title, size=40, pad=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')


predConn_plot(mlp_full_pFC, fc, 'MLP (SC to FC)', os.path.join(root_dir, "Data\\figs\\01_sfc_validation\\mlp_pfc.png"))
predConn_plot(mlp_full_pSC, sc, 'MLP (FC to SC)', os.path.join(root_dir, "Data\\figs\\01_sfc_validation\\mlp_psc.png"))
predConn_plot(pgcn_full_pFC, fc, 'pGCN (SC to FC)', os.path.join(root_dir, "Data\\figs\\01_sfc_validation\\pgcn_pfc.png"))
predConn_plot(pgcn_full_pSC, sc, 'pGCN (FC to SC)', os.path.join(root_dir, "Data\\figs\\01_sfc_validation\\pgcn_psc.png"))