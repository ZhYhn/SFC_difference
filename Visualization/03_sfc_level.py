import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
import nibabel as nib
import pandas as pd
import h5py
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)


def normalize(matrix, method="min-max"):
    # normalize by row
    if method == "min-max":
        max_vals, min_vals = matrix.max(axis=1, keepdims=True), matrix.min(axis=1, keepdims=True)
        return (matrix - min_vals) / (max_vals - min_vals)
    elif method == "z-score":
        mean_vals, std_vals = matrix.mean(axis=1, keepdims=True), matrix.std(axis=1, keepdims=True)
        return (matrix - mean_vals) / std_vals
    
def load_diff(sfc_dir, normalize_method="min-max"):
    # only informational diff.
    sfc_diff = {
        "Info": {
            "sc2fc": {},
            "fc2sc": {}
        }
    }
    for root, dirs, files in os.walk(sfc_dir):
        for file in files:
            if "full" in file:
                full = np.load(os.path.join(root, file))
                if "sc2fc" in file:
                    direct = np.load(os.path.join(root, file).replace("full", "direct"))
                    sfc_diff["Info"]["sc2fc"][file.split("_")[1]] = normalize(np.abs(full - direct), normalize_method)
                elif "fc2sc" in file:
                    direct = np.load(os.path.join(root, file).replace("full", "direct"))
                    sfc_diff["Info"]["fc2sc"][file.split("_")[1]] = normalize(np.abs(full - direct), normalize_method)
    return sfc_diff

def get_mean(info_diff, total=True):
    info_diff_array = [info_diff[approach] for approach in info_diff]
    n_regions = info_diff_array[0].shape[1]
    mean_per_region = np.zeros(n_regions)
    mean_per_region_split = np.zeros((len(info_diff_array), n_regions))
    for i in range(n_regions):
        feature = np.array([arr[:, i] for arr in info_diff_array])
        if total:
            mean_per_region[i] = feature.mean()
        else:
            mean_per_region_split[:, i] = feature.mean(axis=1)
    if total:
        return mean_per_region
    else:
        return mean_per_region_split

def get_std(info_diff):
    info_diff_array = [info_diff[approach] for approach in info_diff]
    n_regions = info_diff_array[0].shape[1]
    std_per_region = np.zeros(n_regions)
    for i in range(n_regions):
        feature = np.array([arr[:, i] for arr in info_diff_array])
        std_per_region[i] = feature.std(axis=0).mean()
    return std_per_region

info_diff = load_diff(os.path.join(root_dir, "Data\\sfc"))

full_sc_sgcn = np.load(os.path.join(root_dir, "Data\\sfc\\sGCN\\sfc_sgcn_pFC_full.npy"))
full_fc_sgcn = np.load(os.path.join(root_dir, "Data\\sfc\\sGCN\\sfc_sgcn_pSC_full.npy"))
direct_sgcn = np.load(os.path.join(root_dir, "Data\\sfc\\sGCN\\sfc_sgcn_direct.npy"))
info_diff["Info"]["sc2fc"]["sgcn"] = normalize(np.abs(full_sc_sgcn - direct_sgcn), "min-max")
info_diff["Info"]["fc2sc"]["sgcn"] = normalize(np.abs(full_fc_sgcn - direct_sgcn), "min-max")

std_sc2fc = get_std(info_diff["Info"]["sc2fc"])
mean_sc2fc = get_mean(info_diff["Info"]["sc2fc"])

std_fc2sc = get_std(info_diff["Info"]["fc2sc"])
mean_fc2sc = get_mean(info_diff["Info"]["fc2sc"])



def aggregation(vector):
    tab = pd.read_excel(os.path.join(root_dir, "others\\allTables.xlsx"), sheet_name="table 2")
    transf = pd.concat([tab.iloc[:, [0, 2]], tab.iloc[:, [4, 6]].rename(columns={'Idx..1':'Idx.', 'Orig..1':'Orig.'}), tab.iloc[:, [8, 10]].rename(columns={'Idx..2':'Idx.', 'Orig..2':'Orig.'})], axis=0)
    transf = transf.sort_values(by='Orig.').reset_index(drop=True)
    vector_new = np.zeros_like(vector)
    for i in range(360):
        idx = transf.iloc[i, 0]-1 if i < 180 else transf.iloc[i-180, 0]-1+180
        vector_new[idx] = vector[i]
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
    result = np.zeros(20)
    for idx, boundary in enumerate(boundaries):
        start, end = boundary
        result[idx] = vector_new[start:end].mean()
    return result

network_names = ['L Visual','L Somatomotor','L Cingulo-opercular','L Dorsal attention','L Language',
                 'L Frontoparietal','L Auditory','L Default mode','L Multimodal','L Orbito-affective',
                 'R Visual','R Somatomotor','R Cingulo-opercular','R Dorsal attention','R Language',
                 'R Frontoparietal','R Auditory','R Default mode','R Multimodal','R Orbito-affective']

plt.figure(figsize=(12,4))
indices = np.argsort(aggregation(mean_sc2fc))
plt.bar(np.arange(20), aggregation(mean_sc2fc)[indices])
plt.title("SC to FC", fontsize=15)
plt.yticks(fontsize=10)
plt.xticks(np.arange(20), np.array(network_names)[indices], fontsize=10, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(root_dir, "Data\\figs\\03_sfc_level\\infoDiff_sorted_sc2fc.png"), dpi=600)

plt.figure(figsize=(12,4))
indices = np.argsort(aggregation(mean_fc2sc))
plt.bar(np.arange(20), aggregation(mean_fc2sc)[indices])
plt.title("FC to SC", fontsize=15)
plt.yticks(fontsize=10)
plt.xticks(np.arange(20), np.array(network_names)[indices], fontsize=10, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(root_dir, "Data\\figs\\03_sfc_level\\infoDiff_sorted_fc2sc.png"), dpi=600)



def scatter_plot(std, mean, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    colors_hex = ['#dedcdc', '#f7b99e', '#df624e', '#b50727']
    positions = [0.0, 0.25, 0.5, 1.0]
    colors_rgb = [mcolors.hex2color(c) for c in colors_hex]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list('workbench_warm', list(zip(positions, colors_rgb)), N=256)
    mean_max = mean.max()
    mean_norm = np.clip(mean / mean_max, 0, 1)
    colors = cmap_custom(mean_norm)

    ax.scatter(std, mean, c=colors, alpha=0.7, edgecolors='none')
    sns.regplot(x=std, y=mean, ax=ax, scatter=False, color='gray', line_kws={'linewidth':2})
    
    r, p = pearsonr(std, mean)
    textstr = f'r = {r:.3f}\n{p_text(p)}'
    ax.text(0.04, 0.85, textstr, transform=ax.transAxes, fontsize=15,
            verticalalignment='bottom', horizontalalignment='left')
    
    ax.set_xlabel('Cross-approach standard deviation', fontsize=20, labelpad=5)
    plt.xticks(fontsize=15)
    ax.set_ylabel('Cross-approach mean', fontsize=20, labelpad=5)
    plt.yticks(fontsize=15)
    plt.title(title, size=25, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show

def p_text(p):
    if p < 0.001:
        return "p < 0.001"
    else:
        return f"p={p:.3f}"
    
scatter_plot(std_sc2fc, mean_sc2fc, "SC to FC", os.path.join(root_dir, "Data\\figs\\03_sfc_level\\info_axis_sc2fc.png"))
scatter_plot(std_fc2sc, mean_fc2sc, "FC to SC", os.path.join(root_dir, "Data\\figs\\03_sfc_level\\info_axis_fc2sc.png"))



def load_sfc(path):
    sfc = {
        "sc2fc": {
            "direct":{},
            "semi-indirect":{},
            "indirect":{}
        },
        "fc2sc": {
            "direct":{},
            "semi-indirect":{},
            "indirect":{}
        }
    }
    
    for root, dirs, files in os.walk(path):
        for file in files:
            array = np.load(os.path.join(root, file))
            if "sc2fc" in file:
                direction = "sc2fc"
            elif "fc2sc" in file:
                direction = "fc2sc"
            else:
                if "corr" in file:
                    sfc["sc2fc"]["direct"][file] = array
                    sfc["fc2sc"]["direct"][file] = array
                elif "sgcn_direct" in file:
                    sfc["sc2fc"]["direct"][file] = array
                    sfc["fc2sc"]["direct"][file] = array
                elif "sgcn_full" in file:
                    sfc["sc2fc"]["indirect"][file] = array
                    sfc["fc2sc"]["indirect"][file] = array
                elif "sgcn_pFC_full" in file:
                    sfc["sc2fc"]["semi-indirect"][file] = array
                elif "sgcn_pSC_full" in file:
                    sfc["fc2sc"]["semi-indirect"][file] = array
            if "full" in file:
                sfc[direction]["semi-indirect"][file] = array
            elif "direct" in file:
                sfc[direction]["direct"][file] = array
    return sfc

def pca_scatter(sfc, direction, title, save_path):
    n_sample = 400
    to_marker = {"direct":"^", "semi-indirect":"o", "indirect":"s"}
    to_color = {'Correlation':'#1f77b4', 'Regression':'#d62728', 'MLP':'#2ca02c', 'pGCN':'#ff7f0e', 'sGCN':'#9467bd'}
    to_label = {'sfc_corr.npy':'Correlation', 
                'sfc_regr_sc2fc_full.npy':'full Regression', 'sfc_regr_sc2fc_direct.npy':'direct Regression', 
                'sfc_regr_fc2sc_full.npy':'full Regression', 'sfc_regr_fc2sc_direct.npy':'direct Regression', 
                'sfc_mlp_sc2fc_full.npy':'full MLP', 'sfc_mlp_sc2fc_direct.npy':'direct MLP', 
                'sfc_mlp_fc2sc_full.npy':'full MLP', 'sfc_mlp_fc2sc_direct.npy':'direct MLP', 
                'sfc_pgcn_sc2fc_full.npy':'full pGCN', 'sfc_pgcn_sc2fc_direct.npy':'direct pGCN', 
                'sfc_pgcn_fc2sc_full.npy':'full pGCN', 'sfc_pgcn_fc2sc_direct.npy':'direct pGCN', 
                'sfc_sgcn_pFC_full.npy':'full-SC sGCN', 'sfc_sgcn_direct.npy':'direct sGCN', 
                'sfc_sgcn_pSC_full.npy':'full-FC sGCN', 'sfc_sgcn_full.npy':'full sGCN'}
    
    arrs_direct, labels_direct, infoCombs_direct = get_arrs_labels(sfc, direction, "direct", to_label, n_sample)
    arrs_semi_indirect, labels_semi_indirect, infoCombs_semi_indirect = get_arrs_labels(sfc, direction, "semi-indirect", to_label, n_sample)
    arrs_indirect, labels_indirect, infoCombs_indirect = get_arrs_labels(sfc, direction, "indirect", to_label, n_sample)
    
    arrs = np.vstack(arrs_direct + arrs_semi_indirect + arrs_indirect)
    labels = np.array(labels_direct + labels_semi_indirect + labels_indirect)
    infoCombs = np.array(infoCombs_direct + infoCombs_semi_indirect + infoCombs_indirect)
    
    pca = PCA(n_components=2)
    pca.fit(arrs)
    pca_arrs = pca.transform(arrs)
    pc1, pc2 = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6))
    for infoComb in ["direct", "semi-indirect", "indirect"]:
        mask = infoCombs == infoComb
        label_mask = np.array([label.split(" ")[-1] for label in labels[mask]])
        for a in ['Correlation', 'Regression', 'MLP', 'pGCN', 'sGCN']:
            if len(labels[mask][label_mask == a]) != 0:
                plt.scatter(pca_arrs[mask, 0][label_mask == a], pca_arrs[mask, 1][label_mask == a], 
                            c=to_color[a], 
                            label=labels[mask][label_mask == a][0], alpha=0.6, s=20, marker=to_marker[infoComb])
    plt.xlabel(f'PC1 ({pc1:.2%})', fontsize=20, labelpad=5)
    plt.xticks(fontsize=15)
    plt.ylabel(f'PC2 ({pc2:.2%})', fontsize=20, labelpad=5)
    plt.yticks(fontsize=15)
    plt.title(title, size=25, pad=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
def get_arrs_labels(sfc, direction, infoComb, to_label, n_sample):
    arrs = []
    labels = []
    infoCombs = []
    for approach, arr in sfc[direction][infoComb].items():
        arrs.append(arr)
        labels += [to_label[approach]]*n_sample
        infoCombs += [infoComb]*n_sample
    return arrs, labels, infoCombs

sfc = load_sfc(os.path.join(root_dir, "Data\\sfc"))
pca_scatter(sfc, "sc2fc", "SC to FC", os.path.join(root_dir, r"Data\figs\03_sfc_level\pca_sc2fc.png"))
pca_scatter(sfc, "fc2sc", "FC to SC", os.path.join(root_dir, r"Data\figs\03_sfc_level\pca_fc2sc.png"))



def violinplot(direct, full_sc, full_fc, full, save_path, reference_lines=None):
    plt.figure(figsize=(10, 6))
    data = {
        'value': np.concatenate([direct.ravel(), full_sc.ravel(), full_fc.ravel(), full.ravel()]),
        'group': ['direct sGCN'] * direct.size + ['full-SC sGCN'] * full_sc.size + ['full-FC sGCN'] * full_fc.size + ['full sGCN'] * full.size
    }
    df = pd.DataFrame(data)
    ax = sns.violinplot(x='group', y='value', data=df, color='#9467bd', alpha=0.6, width=0.4)
    plt.xlabel("")
    plt.xticks(fontsize=20)
    plt.ylabel("Structure-function coupling", fontsize=20, labelpad=5)
    plt.yticks(fontsize=15)

    if reference_lines:
        for line in reference_lines:
            ax.axhline(y=line['y'], color=line['color'], linestyle=line.get('linestyle', '--'),
                       alpha=line.get('alpha', 1.0), label=line['label'])
    if reference_lines:
        ax.legend(loc='lower right',fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

direct_sgcn = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_direct.npy"))
full_sc_sgcn = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_pFC_full.npy"))
full_fc_sgcn = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_pSC_full.npy"))
full_sgcn = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_full.npy"))

corr = np.median(np.load(os.path.join(root_dir, r"Data\sfc\Corr\sfc_corr.npy")))
full_regr_sc2fc = np.median(np.load(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_sc2fc_full.npy")))
full_mlp_sc2fc = np.median(np.load(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_sc2fc_full.npy")))
full_pgcn_sc2fc = np.median(np.load(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_sc2fc_full.npy")))

full_regr_fc2sc = np.median(np.load(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_fc2sc_full.npy")))
full_mlp_fc2sc = np.median(np.load(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_fc2sc_full.npy")))
full_pgcn_fc2sc = np.median(np.load(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_fc2sc_full.npy")))

color_corr = '#1f77b4'
color_regr = '#d62728'
color_mlp = '#2ca02c'
color_pgcn = '#ff7f0e'
alpha_common = 0.6

reference_lines = [
    {'y': corr, 'label': 'Correlation', 'color': color_corr, 'linestyle': '-', 'alpha': alpha_common},
    {'y': full_regr_sc2fc, 'label': 'Regression (SC to FC)', 'color': color_regr, 'linestyle': '-', 'alpha': alpha_common},
    {'y': full_mlp_sc2fc, 'label': 'MLP (SC to FC)', 'color': color_mlp, 'linestyle': '-', 'alpha': alpha_common},
    {'y': full_pgcn_sc2fc, 'label': 'pGCN (SC to FC)', 'color': color_pgcn, 'linestyle': '-', 'alpha': alpha_common},
    {'y': full_regr_fc2sc, 'label': 'Regression (FC to SC)', 'color': color_regr, 'linestyle': '--', 'alpha': alpha_common},
    {'y': full_mlp_fc2sc, 'label': 'MLP (FC to SC)', 'color': color_mlp, 'linestyle': '--', 'alpha': alpha_common},
    {'y': full_pgcn_fc2sc, 'label': 'pGCN (FC to SC)', 'color': color_pgcn, 'linestyle': '--', 'alpha': alpha_common},
]

violinplot(direct_sgcn, full_sc_sgcn, full_fc_sgcn, full_sgcn, os.path.join(root_dir, r"Data\figs\03_sfc_level\sfc_sgcn.png"), reference_lines)