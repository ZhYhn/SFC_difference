import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as mcolors


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

regr_diff_i_sc = np.load(os.path.join(root_dir, r"Data\difference\Regr\diff_i_sc.npy"))
regr_diff_i_fc = np.load(os.path.join(root_dir, r"Data\difference\Regr\diff_i_fc.npy"))
regr_diff_m_sc = np.load(os.path.join(root_dir, r"Data\difference\Regr\diff_m_sc.npy"))
regr_diff_m_fc = np.load(os.path.join(root_dir, r"Data\difference\Regr\diff_m_fc.npy"))

mlp_diff_i_sc = np.load(os.path.join(root_dir, r"Data\difference\MLP\diff_i_sc.npy"))
mlp_diff_i_fc = np.load(os.path.join(root_dir, r"Data\difference\MLP\diff_i_fc.npy"))
mlp_diff_m_sc = np.load(os.path.join(root_dir, r"Data\difference\MLP\diff_m_sc.npy"))
mlp_diff_m_fc = np.load(os.path.join(root_dir, r"Data\difference\MLP\diff_m_fc.npy"))

pgcn_diff_i_sc = np.load(os.path.join(root_dir, r"Data\difference\pGCN\diff_i_sc.npy"))
pgcn_diff_i_fc = np.load(os.path.join(root_dir, r"Data\difference\pGCN\diff_i_fc.npy"))
pgcn_diff_m_sc = np.load(os.path.join(root_dir, r"Data\difference\pGCN\diff_m_sc.npy"))
pgcn_diff_m_fc = np.load(os.path.join(root_dir, r"Data\difference\pGCN\diff_m_fc.npy"))



def proportion(diff):
    return np.abs(diff[diff < 0].mean()), diff[diff > 0].mean()
def mean(diff):
    return (diff < 0).sum()/diff.size, (diff > 0).sum()/diff.size


def barplot(groups, save_path):
    group_titles = ['Regression', 'MLP', 'pGCN']
    categories = ['Info. (SC to FC)', 'Info. (FC to SC)', 'Meth. (SC to FC)', 'Meth. (FC to SC)']
    colors = ['#507FFF', '#FF7F50']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    bar_width = 0.23
    x = np.arange(len(categories))
    for i, group_data in enumerate(groups):
        ax = axes[i]
        for j, (label, color) in enumerate(zip(['negative', 'positive'], colors)):
            offset = (j - 0.5) * bar_width
            values = [tup[j] for tup in group_data]
            bars = ax.bar(x + offset, values, width=bar_width, color=color,
                          label=label, edgecolor='black', linewidth=0.5)
            ax.bar_label(bars, fmt='%.2f', label_type='edge', padding=2, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=15, rotation=30)
        ax.set_yticks([])
        ax.set_title(group_titles[i], fontsize=25, pad=10)

    axes[0].legend(title='Mean', title_fontsize=13, fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

diffs_prop = [[proportion(regr_diff_i_sc), proportion(regr_diff_i_fc), proportion(regr_diff_m_sc), proportion(regr_diff_m_fc)], 
              [proportion(mlp_diff_i_sc), proportion(mlp_diff_i_fc), proportion(mlp_diff_m_sc), proportion(mlp_diff_m_fc)], 
              [proportion(pgcn_diff_i_sc), proportion(pgcn_diff_i_fc), proportion(pgcn_diff_m_sc), proportion(pgcn_diff_m_fc)]]
barplot(diffs_prop, os.path.join(root_dir, "Data\\figs\\supplement\\S2_diff_prop.png"))
diffs_mean = [[mean(regr_diff_i_sc), mean(regr_diff_i_fc), mean(regr_diff_m_sc), mean(regr_diff_m_fc)], 
              [mean(mlp_diff_i_sc), mean(mlp_diff_i_fc), mean(mlp_diff_m_sc), mean(mlp_diff_m_fc)], 
              [mean(pgcn_diff_i_sc), mean(pgcn_diff_i_fc), mean(pgcn_diff_m_sc), mean(pgcn_diff_m_fc)]]
barplot(diffs_mean, os.path.join(root_dir, "Data\\figs\\supplement\\S2_diff_mean.png"))



def reorder(matrix):
    tab = pd.read_excel(os.path.join(root_dir, r"others\allTables.xlsx"), sheet_name="table 2")
    transf = pd.concat([tab.iloc[:, [0, 2]], tab.iloc[:, [4, 6]].rename(columns={'Idx..1':'Idx.', 'Orig..1':'Orig.'}), tab.iloc[:, [8, 10]].rename(columns={'Idx..2':'Idx.', 'Orig..2':'Orig.'})], axis=0)
    transf = transf.sort_values(by='Orig.').reset_index(drop=True)
    matrix_new = np.zeros_like(matrix)
    for i in range(360):
        for j in range(360):
            y = transf.iloc[i, 0]-1 if i < 180 else transf.iloc[i-180, 0]-1+180
            x = transf.iloc[j, 0]-1 if j < 180 else transf.iloc[j-180, 0]-1+180
            matrix_new[y, x] = matrix[i, j]
    return matrix_new

def diffConn_plot(sc, fc, title, save_path):
    indices = np.triu_indices(sc.shape[0], k=1)
    sc[indices], fc[indices] = 0, 0
    matrix = fc.T + sc
    matrix[np.eye(sc.shape[0], dtype=bool)] = 0
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(reorder(matrix), 
                    cmap='coolwarm', 
                    center=0, 
                    square=True,
                    xticklabels=False,
                    yticklabels=False, 
                    cbar=True)
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
    for boundary in boundaries:
        start, end = boundary
        rect = patches.Rectangle((start, start), end - start, end - start,
                                  linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    for i, (start, end) in enumerate(boundaries, start=1):
        x = start - 5 if i%10 != 9 else start - 7
        y = end + 5 if i%10 != 9 else end + 2
        ax.text(x, y, f'({i})', 
                ha='center', va='center', 
                fontsize=12, color='black')

    plt.xlim(-1, 361)
    plt.ylim(361, -1)
    plt.title(title, size=30, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)


diffConn_plot(regr_diff_i_sc.mean(axis=0), regr_diff_i_fc.mean(axis=0), 'Regression (info.)', 
              os.path.join(root_dir, "Data\\figs\\02_predConn_level\\regr_diff_info.png"))
diffConn_plot(regr_diff_m_sc.mean(axis=0), regr_diff_m_fc.mean(axis=0), 'Regression (meth.)', 
              os.path.join(root_dir, "Data\\figs\\02_predConn_level\\regr_diff_meth.png"))

diffConn_plot(mlp_diff_i_sc.mean(axis=0), mlp_diff_i_fc.mean(axis=0), 'MLP (info.)', 
              os.path.join(root_dir, "Data\\figs\\02_predConn_level\\mlp_diff_info.png"))
diffConn_plot(mlp_diff_m_sc.mean(axis=0), mlp_diff_m_fc.mean(axis=0), 'MLP (meth.)', 
              os.path.join(root_dir, "Data\\figs\\02_predConn_level\\mlp_diff_meth.png"))

diffConn_plot(pgcn_diff_i_sc.mean(axis=0), pgcn_diff_i_fc.mean(axis=0), 'pGCN (info.)', 
              os.path.join(root_dir, "Data\\figs\\02_predConn_level\\pgcn_diff_info.png"))
diffConn_plot(pgcn_diff_m_sc.mean(axis=0), pgcn_diff_m_fc.mean(axis=0), 'pGCN (meth.)', 
              os.path.join(root_dir, "Data\\figs\\02_predConn_level\\pgcn_diff_meth.png"))



h5_path = os.path.join(root_dir, r"others\parcelMyelinationIndices.mat")
with h5py.File(h5_path, "r") as f:
    mi = f['myelinationIndices'][0]


def scatter_plot(diff, diff2, mi, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    r1, p1 = pearsonr(mi, diff)
    sns.regplot(x=mi, y=diff, ax=ax, color='#507FFF', line_kws={'linewidth':2}, label=f'SC to FC (r={r1:.3f}, {p_text(p1)})')
    r2, p2 = pearsonr(mi, diff2)
    sns.regplot(x=mi, y=diff2, ax=ax, color='#00008B', line_kws={'linewidth':2}, label=f'FC to SC (r={r2:.3f}, {p_text(p2)})')
    
    ax.set_xlabel('Myelination index (T1w/T2w)', fontsize=25, labelpad=15)
    plt.xticks(fontsize=20)
    ax.set_ylabel('Absolute informational difference', fontsize=25, labelpad=10)
    plt.yticks(fontsize=20)
    plt.title(title, size=30, pad=15)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
def p_text(p):
    if p < 0.001:
        return "p<0.001"
    else:
        return f"p={p:.3f}"
def preprocess(diff):
    diff[:, np.eye(diff.shape[1], dtype=bool)] = 0
    result = np.abs(diff).copy()
    return result.sum(axis=1).mean(axis=0)

scatter_plot(preprocess(mlp_diff_i_sc), preprocess(mlp_diff_i_fc), mi, "MLP", 
             os.path.join(root_dir, "Data\\figs\\02_predConn_level\\mlp_myelin_info.png"))

scatter_plot(preprocess(regr_diff_i_sc), preprocess(regr_diff_i_fc), mi, "Regression", 
             os.path.join(root_dir, "Data\\figs\\02_predConn_level\\regr_myelin_info.png"))

scatter_plot(preprocess(pgcn_diff_i_sc), preprocess(pgcn_diff_i_fc), mi, "pGCN", 
             os.path.join(root_dir, "Data\\figs\\02_predConn_level\\pgcn_myelin_info.png"))