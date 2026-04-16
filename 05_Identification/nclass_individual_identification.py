import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import os
import pickle
from tqdm import tqdm
import warnings


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)


def get_dataset(sfc, n_class):
    random_indices = np.random.choice(100, size=n_class, replace=False)
    random_indices = np.sort(np.hstack([random_indices*4, random_indices*4+1, random_indices*4+2, random_indices*4+3]))
    sfc = sfc[random_indices, :]
    subjs = [subj for subj in range(n_class) for _ in range(4)]
    subjs = np.array(subjs)
    region_names = np.load(os.path.join(root_dir, r"others\HCP-MMP1_regions.npy"), allow_pickle=True)
    sfc_dict = {'subject': subjs}
    for i, col_name in enumerate(region_names):
        sfc_dict[col_name] = sfc[:, i]
    return pd.DataFrame(sfc_dict)

def lastN_classification(sfc, n_class, method):
    df = get_dataset(sfc, n_class)
    df.iloc[:, 1:] = 2 * df.iloc[:, 1:].values - 1
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

    if method == 'svm':
        model = svm.SVC(kernel='rbf')
    elif method == 'rf':
        model = RandomForestClassifier(n_estimators=100)
    elif method == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=3000)
    
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=20, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = cross_val_score(model, X, y, cv=skf)
    # print(f"{method}:")
    # print(f"average acc: {scores.mean():.4f} (±{scores.std():.4f})")
    # print()
    return scores.mean() # (scores.mean(), scores.std())



corr = np.load(os.path.join(root_dir, r"Data\sfc\Corr\sfc_corr.npy"))

regr_sc2fc_full = np.load(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_sc2fc_full.npy"))
regr_sc2fc_direct = np.load(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_sc2fc_direct.npy"))
regr_fc2sc_full = np.load(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_fc2sc_full.npy"))
regr_fc2sc_direct = np.load(os.path.join(root_dir, r"Data\sfc\Regr\sfc_regr_fc2sc_direct.npy"))

mlp_sc2fc_full = np.load(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_sc2fc_full.npy"))
mlp_sc2fc_direct = np.load(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_sc2fc_direct.npy"))
mlp_fc2sc_full = np.load(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_fc2sc_full.npy"))
mlp_fc2sc_direct = np.load(os.path.join(root_dir, r"Data\sfc\MLP\sfc_mlp_fc2sc_direct.npy"))

pgcn_sc2fc_full = np.load(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_sc2fc_full.npy"))
pgcn_sc2fc_direct = np.load(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_sc2fc_direct.npy"))
pgcn_fc2sc_full = np.load(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_fc2sc_full.npy"))
pgcn_fc2sc_direct = np.load(os.path.join(root_dir, r"Data\sfc\pGCN\sfc_pgcn_fc2sc_direct.npy"))

sgcn_full = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_full.npy"))
sgcn_pFC_full = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_pFC_full.npy"))
sgcn_pSC_full = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_pSC_full.npy"))
sgcn_direct = np.load(os.path.join(root_dir, r"Data\sfc\sGCN\sfc_sgcn_direct.npy"))


results = {
    'SVC':{
        'Corr':{
            'full':[]
        }, 
        'Regr':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'MLP':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'pGCN':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'sGCN':{
            'full':[],
            'SC_full':[],
            'FC_full':[],
            'direct':[]
        }
    }, 
    'RF':{
        'Corr':{
            'full':[]
        }, 
        'Regr':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'MLP':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'pGCN':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'sGCN':{
            'full':[],
            'SC_full':[],
            'FC_full':[],
            'direct':[]
        }
    }, 
    'MLP':{
        'Corr':{
            'full':[]
        }, 
        'Regr':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'MLP':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'pGCN':{
            'sc2fc_full':[],
            'sc2fc_direct':[],
            'fc2sc_full':[],
            'fc2sc_direct':[]
        },
        'sGCN':{
            'full':[],
            'SC_full':[],
            'FC_full':[],
            'direct':[]
        }
    }
}


for n_class in tqdm(range(10, 101), total=91):
    corr_svc = lastN_classification(corr, n_class, method='svm')
    corr_rf = lastN_classification(corr, n_class, method='rf')
    corr_mlp = lastN_classification(corr, n_class, method='mlp')
    results['SVC']['Corr']['full'].append(corr_svc)
    results['RF']['Corr']['full'].append(corr_rf)
    results['MLP']['Corr']['full'].append(corr_mlp)

    results['SVC']['Regr']['sc2fc_full'].append(lastN_classification(regr_sc2fc_full, n_class, method='svm'))
    results['RF']['Regr']['sc2fc_full'].append(lastN_classification(regr_sc2fc_full, n_class, method='rf'))
    results['MLP']['Regr']['sc2fc_full'].append(lastN_classification(regr_sc2fc_full, n_class, method='mlp'))
    results['SVC']['Regr']['sc2fc_direct'].append(lastN_classification(regr_sc2fc_direct, n_class, method='svm'))
    results['RF']['Regr']['sc2fc_direct'].append(lastN_classification(regr_sc2fc_direct, n_class, method='rf'))
    results['MLP']['Regr']['sc2fc_direct'].append(lastN_classification(regr_sc2fc_direct, n_class, method='mlp'))
    results['SVC']['Regr']['fc2sc_full'].append(lastN_classification(regr_fc2sc_full, n_class, method='svm'))
    results['RF']['Regr']['fc2sc_full'].append(lastN_classification(regr_fc2sc_full, n_class, method='rf'))
    results['MLP']['Regr']['fc2sc_full'].append(lastN_classification(regr_fc2sc_full, n_class, method='mlp'))
    results['SVC']['Regr']['fc2sc_direct'].append(lastN_classification(regr_fc2sc_direct, n_class, method='svm'))
    results['RF']['Regr']['fc2sc_direct'].append(lastN_classification(regr_fc2sc_direct, n_class, method='rf'))
    results['MLP']['Regr']['fc2sc_direct'].append(lastN_classification(regr_fc2sc_direct, n_class, method='mlp'))

    results['SVC']['MLP']['sc2fc_full'].append(lastN_classification(mlp_sc2fc_full, n_class, method='svm'))
    results['RF']['MLP']['sc2fc_full'].append(lastN_classification(mlp_sc2fc_full, n_class, method='rf'))
    results['MLP']['MLP']['sc2fc_full'].append(lastN_classification(mlp_sc2fc_full, n_class, method='mlp'))
    results['SVC']['MLP']['sc2fc_direct'].append(lastN_classification(mlp_sc2fc_direct, n_class, method='svm'))
    results['RF']['MLP']['sc2fc_direct'].append(lastN_classification(mlp_sc2fc_direct, n_class, method='rf'))
    results['MLP']['MLP']['sc2fc_direct'].append(lastN_classification(mlp_sc2fc_direct, n_class, method='mlp'))
    results['SVC']['MLP']['fc2sc_full'].append(lastN_classification(mlp_fc2sc_full, n_class, method='svm'))
    results['RF']['MLP']['fc2sc_full'].append(lastN_classification(mlp_fc2sc_full, n_class, method='rf'))
    results['MLP']['MLP']['fc2sc_full'].append(lastN_classification(mlp_fc2sc_full, n_class, method='mlp'))
    results['SVC']['MLP']['fc2sc_direct'].append(lastN_classification(mlp_fc2sc_direct, n_class, method='svm'))
    results['RF']['MLP']['fc2sc_direct'].append(lastN_classification(mlp_fc2sc_direct, n_class, method='rf'))
    results['MLP']['MLP']['fc2sc_direct'].append(lastN_classification(mlp_fc2sc_direct, n_class, method='mlp'))

    results['SVC']['pGCN']['sc2fc_full'].append(lastN_classification(pgcn_sc2fc_full, n_class, method='svm'))
    results['RF']['pGCN']['sc2fc_full'].append(lastN_classification(pgcn_sc2fc_full, n_class, method='rf'))
    results['MLP']['pGCN']['sc2fc_full'].append(lastN_classification(pgcn_sc2fc_full, n_class, method='mlp'))
    results['SVC']['pGCN']['sc2fc_direct'].append(lastN_classification(pgcn_sc2fc_direct, n_class, method='svm'))
    results['RF']['pGCN']['sc2fc_direct'].append(lastN_classification(pgcn_sc2fc_direct, n_class, method='rf'))
    results['MLP']['pGCN']['sc2fc_direct'].append(lastN_classification(pgcn_sc2fc_direct, n_class, method='mlp'))
    results['SVC']['pGCN']['fc2sc_full'].append(lastN_classification(pgcn_fc2sc_full, n_class, method='svm'))
    results['RF']['pGCN']['fc2sc_full'].append(lastN_classification(pgcn_fc2sc_full, n_class, method='rf'))
    results['MLP']['pGCN']['fc2sc_full'].append(lastN_classification(pgcn_fc2sc_full, n_class, method='mlp'))
    results['SVC']['pGCN']['fc2sc_direct'].append(lastN_classification(pgcn_fc2sc_direct, n_class, method='svm'))
    results['RF']['pGCN']['fc2sc_direct'].append(lastN_classification(pgcn_fc2sc_direct, n_class, method='rf'))
    results['MLP']['pGCN']['fc2sc_direct'].append(lastN_classification(pgcn_fc2sc_direct, n_class, method='mlp'))


    results['SVC']['sGCN']['full'].append(lastN_classification(sgcn_full, n_class, method='svm'))
    results['RF']['sGCN']['full'].append(lastN_classification(sgcn_full, n_class, method='rf'))
    results['MLP']['sGCN']['full'].append(lastN_classification(sgcn_full, n_class, method='mlp'))
    results['SVC']['sGCN']['SC_full'].append(lastN_classification(sgcn_pFC_full, n_class, method='svm'))
    results['RF']['sGCN']['SC_full'].append(lastN_classification(sgcn_pFC_full, n_class, method='rf'))
    results['MLP']['sGCN']['SC_full'].append(lastN_classification(sgcn_pFC_full, n_class, method='mlp'))
    results['SVC']['sGCN']['FC_full'].append(lastN_classification(sgcn_pSC_full, n_class, method='svm'))
    results['RF']['sGCN']['FC_full'].append(lastN_classification(sgcn_pSC_full, n_class, method='rf'))
    results['MLP']['sGCN']['FC_full'].append(lastN_classification(sgcn_pSC_full, n_class, method='mlp'))
    results['SVC']['sGCN']['direct'].append(lastN_classification(sgcn_direct, n_class, method='svm'))
    results['RF']['sGCN']['direct'].append(lastN_classification(sgcn_direct, n_class, method='rf'))
    results['MLP']['sGCN']['direct'].append(lastN_classification(sgcn_direct, n_class, method='mlp'))



    with open(os.path.join(root_dir, r"Data\identification\nclass_identification.pkl"), 'wb') as f:
        pickle.dump(results, f)