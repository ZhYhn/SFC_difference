import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

with open(os.path.join(root_dir, "Data\\identification\\nclass_identification.pkl"), "rb") as f:
    nclass_identification = pickle.load(f)

with open(os.path.join(root_dir, "Data\\identification\\noisy_identification.pkl"), "rb") as f:
    noisy_identification = pickle.load(f)

def unzip(d):
    new_d = {}
    for key1, value1 in d.items():
        for key2, value2 in value1.items():
            new_d[key1 + ' ' + key2] = value2
    keys = [t[0] for t in new_d.items()]
    vals = [t[1] for t in new_d.items()]
    return keys, vals


def nclass_identification_plot(d, titles, filename):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for idx, classifier in enumerate(['SVC', 'RF', 'MLP']):
        keys, vals = unzip(d[classifier])
    
        prefixes = [key.split()[0] for key in keys]
        unique_groups = list(set(prefixes))
        
        group_to_color = {'Corr':'#1f77b4', 'Regr':'#d62728', 'MLP':'#2ca02c', 'pGCN':'#ff7f0e', 'sGCN':'#9467bd'}
        
        parts = axes[idx].violinplot(vals, positions=range(1, len(keys)+1),
                            showmeans=False, showmedians=True)
        
        for i, body in enumerate(parts['bodies']):
            group = prefixes[i]
            body.set_facecolor(group_to_color[group])
            body.set_alpha(0.7)
            body.set_edgecolor('black')
        
        axes[idx].set_title(titles[idx], fontsize=25, pad=15)
        if idx == 1:
            axes[idx].set_ylabel('Accuracy', fontsize=20, labelpad=15)
        axes[idx].tick_params(axis='y', labelsize=15)
        axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
        if idx == 2:
            axes[idx].set_xticks(range(1, len(keys)+1), ['Correlation',
                                            'Regression (SC to FC)', 'direct Regression (SC to FC)', 'Regression (FC to SC)', 'direct Regression (FC to SC)',
                                            'MLP (SC to FC)', 'direct MLP (SC to FC)', 'MLP (FC to SC)', 'direct MLP (FC to SC)',
                                            'pGCN (SC to FC)', 'direct pGCN (SC to FC)', 'pGCN (FC to SC)', 'direct pGCN (FC to SC)',
                                            'full sGCN', 'full-SC sGCN', 'full-FC sGCN', 'direct sGCN'], rotation=90)
            axes[idx].tick_params(axis='x', labelsize=15)
        else:
            axes[idx].set_xticks([])
    
    legend_patches = [mpatches.Patch(color=group_to_color[group], label='Correlation' if group == 'Corr' else 'Regression' if group == 'Regr' else group)
                      for group in unique_groups]
    axes[0].legend(handles=legend_patches, fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, f"Data\\figs\\04_identification\\{filename}.png"), dpi=600)


nclass_identification_plot(nclass_identification, ["Support vector classifier", "Random forest", "Multilayer perceptron"], "nclass_identification")