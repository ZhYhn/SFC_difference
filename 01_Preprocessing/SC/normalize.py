import numpy as np
import os



def normalize(conn_type):

    # total_sum = 0
    # total_sq_sum = 0
    # total_count = 0
    # for root, _, files in os.walk(os.path.join(r"D:\Download\SFC\SFC_HCP\2_Preprocess\3_intersection\originalDataset", f"{conn_type}")):
    #     for file in files:
    #         path_conn = os.path.join(root, file)
    #         matrix = np.load(path_conn)
    #         if conn_type == 'sc':
    #             matrix = np.log(matrix + 1e-8)
    #         total_sum += np.sum(matrix)
    #         total_sq_sum += np.sum(matrix ** 2)
    #         total_count += matrix.size
    # mean = total_sum / total_count
    # std = np.sqrt(total_sq_sum / total_count - mean ** 2)

    # print(f"Mean: {mean}, Std: {std}")

    for root, _, files in os.walk(os.path.join(r"D:\Download\SFC\SFC_HCP\2_Preprocess\3_intersection\originalDataset", f"{conn_type}")):
        for file in files:
            path_conn = os.path.join(root, file)
            matrix = np.load(path_conn)
            if conn_type == 'sc':
                lg_matrix = np.log(matrix + 1e-8)
                mask = ~np.eye(lg_matrix.shape[0], dtype=bool)
                off_diag = lg_matrix[mask]
                matrix = (lg_matrix - off_diag.min()) / (off_diag.max() - off_diag.min())
            # normalized_matrix = (matrix - mean) / std
            save_dir = os.path.join(r"D:\Download\SFC\SFC_HCP\2_Preprocess\4_Normalization\normalizedDataset", f"{conn_type}")
            # np.save(os.path.join(save_dir, file.replace('.npy', '_normalized.npy')), normalized_matrix)
            if conn_type == 'sc':
                np.save(os.path.join(save_dir, file.replace('.npy', '_log01.npy')), matrix)
            else:
                np.save(os.path.join(save_dir, file), matrix)


if __name__ == "__main__":
    normalize('sc')