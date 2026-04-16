import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)


def load_diff(sfc_path):
    sfc_diff = {
        "info": {
            "sc2fc": {},
            "fc2sc": {}
        }, 
        "meth": {
            "sc2fc": {},
            "fc2sc": {}
        }
    }
    for root, dirs, files in os.walk(sfc_path):
        for file in files:
            if "full" in file:
                full = np.load(os.path.join(root, file))
                if "sc2fc" in file:
                    direct = np.load(os.path.join(root, file).replace("full", "direct"))
                    sfc_diff["info"]["sc2fc"][file.split("_")[1]] = np.abs(full - direct)
                elif "fc2sc" in file:
                    direct = np.load(os.path.join(root, file).replace("full", "direct"))
                    sfc_diff["info"]["fc2sc"][file.split("_")[1]] = np.abs(full - direct)
                elif "sgcn_pFC_full" in file:
                    direct = np.load(os.path.join(root, "sfc_sgcn_direct.npy"))
                    sfc_diff["info"]["sc2fc"]["sgcn"] = np.abs(full - direct)
                elif "sgcn_pSC_full" in file:
                    direct = np.load(os.path.join(root, "sfc_sgcn_direct.npy"))
                    sfc_diff["info"]["fc2sc"]["sgcn"] = np.abs(full - direct)
                    
            elif "direct" in file:
                corr = np.load(os.path.join(sfc_path, r"Corr\sfc_corr.npy"))
                direct = np.load(os.path.join(root, file))
                if "sc2fc" in file:
                    sfc_diff["meth"]["sc2fc"][file.split("_")[1]] = np.abs(direct - corr)
                elif "fc2sc" in file:
                    sfc_diff["meth"]["fc2sc"][file.split("_")[1]] = np.abs(direct - corr)
                else:
                    sfc_diff["meth"]["sc2fc"]["sgcn"] = np.abs(direct - corr)
                    sfc_diff["meth"]["fc2sc"]["sgcn"] = np.abs(direct - corr)
    return sfc_diff
    
diff = load_diff(os.path.join(root_dir, "Data\\sfc"))

n_nodes = 360
all_tstats, all_pvals = [], []
diff_src = []
for direction in ["sc2fc", "fc2sc"]:
    for approach in ["regr", "mlp", "pgcn", "sgcn"]:
        info = diff["info"][direction][approach]
        meth = diff["meth"][direction][approach]
        for node in range(n_nodes):
            data = meth[:, node] - info[:, node]
            t_stat, p_val = stats.ttest_1samp(data, 0)
            all_tstats.append(t_stat)
            all_pvals.append(p_val)
            diff_src.append(f"{approach}_{direction}")
reject, _, _, _ = multipletests(all_pvals, alpha=0.01, method='fdr_bh')
ratios = {}
for src in set(diff_src):
    ratios[src] = [0, 0, 0] # [pos:meth>info, neg:info>meth, significant_nums]
for t, src, rej in zip(all_tstats, diff_src, reject):
    if rej:
        if t > 0:
            ratios[src][0] += 1/n_nodes
        elif t < 0:
            ratios[src][1] += 1/n_nodes
        ratios[src][2] += 1

print(ratios)