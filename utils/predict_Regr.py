import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import expm
import os
from sklearn.linear_model import LinearRegression
from tqdm import tqdm



def predict_regr(sc, fc, single_node=None):
    '''
    Input arrays must be 2D.
    '''
    sc, fc = sc.copy().astype(np.float32), fc.copy().astype(np.float32)

    sc[sc != 0] = 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    centroids = np.load(os.path.join(root_dir, "others\\HCP_MMP1_centroids.npy"))
    euclidean_distance = cdist(centroids, centroids, metric='euclidean')
    mask = np.eye(360, dtype=bool)
    euclidean_distance = (euclidean_distance - euclidean_distance[~mask].mean()) / euclidean_distance[~mask].std()

    path_length, communicability = getX(sc)
    if single_node != None:
        return get_pFC(path_length, communicability, euclidean_distance, fc, single_node)[0]
    else:
        pFC = np.zeros((sc.shape[0], sc.shape[0]))
        for node_i in range(sc.shape[0]):
            pFC[node_i] = get_pFC(path_length, communicability, euclidean_distance, fc, node_i)
        return pFC



def get_pFC(path_length, communicability, euclidean_distance, fc, node_i):

    y_train = np.delete(fc[node_i, :], node_i)
    X_euclidean_train = np.delete(euclidean_distance[node_i, :], node_i)
    X_path_train = np.delete(path_length[node_i, :], node_i)
    X_comm_train = np.delete(communicability[node_i, :], node_i)
    
    X_train = np.column_stack([X_euclidean_train, X_path_train, X_comm_train])
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    return np.expand_dims(np.insert(y_pred, node_i, 1), axis=0)



def getX(sc):
    mask = np.eye(sc.shape[0], dtype=bool)
    path_length = shortest_path(sc, directed=False, method='D', unweighted=True)
    path_length[path_length == np.inf] = 361
    path_length = (path_length - path_length[~mask].mean()) / path_length[~mask].std()
    max_degree = np.max(np.sum(sc, axis=1))
    communicability = np.log(expm(sc / max_degree) + 1e-8)
    communicability = (communicability - communicability[~mask].mean()) / communicability[~mask].std()
    return path_length, communicability