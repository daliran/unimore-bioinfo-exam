import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def create_dataset_masks(
    points: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
):
    total = train_split + val_split + test_split

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {total}")

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(points, generator=generator)
    else:
        perm = torch.randperm(points)

    num_train = int(train_split * points)
    num_val = int(val_split * points)
    #num_test = points - num_train - num_val

    train_idx = perm[:num_train]
    val_idx = perm[num_train:num_train + num_val]
    test_idx = perm[num_train + num_val:]

    train_mask = torch.zeros(points, dtype=torch.bool)
    val_mask = torch.zeros(points, dtype=torch.bool)
    test_mask = torch.zeros(points, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

def merge_dataset_masks(mask1, mask2):
    combined = mask1 | mask2
    return combined

def load_raw_data(file_path):
    df = pd.read_csv(file_path)

    label_column = df.iloc[:, 0]
    raw_labels = torch.tensor(label_column.str.strip() == "Luminal B", dtype=torch.long)

    feature_df = df.iloc[:, 1:]
    raw_features = torch.tensor(feature_df.values, dtype=torch.float64)

    return raw_features, raw_labels

def remove_null_features(raw_features):
    # remove features with 0 values
    mask = (raw_features != 0).any(dim=0)
    filtered_features = raw_features[:, mask]
    return filtered_features

def normalize_features(features, target_dims, seed, use_log: bool, use_standard_scaler: bool, use_pca: bool):

    normalized_data = features.numpy()

    if use_log:
        eps = np.finfo(float).eps
        normalized_data = np.log2(normalized_data + eps)

    if use_standard_scaler:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(normalized_data)
        normalized_data = scaled_data

    if use_pca:
        if seed is not None:
            pca = PCA(n_components=target_dims, random_state=seed)
        else:
            pca = PCA(n_components=target_dims)

        reduced_data = pca.fit_transform(normalized_data)
        normalized_data = reduced_data

    return torch.tensor(normalized_data, dtype=torch.float32)

def compute_abs_pearson_correlation(features: torch.Tensor):
    df_temp = pd.DataFrame(features.numpy())
    pearson_corr = df_temp.T.corr(method="pearson")
    pearson_coor_abs = pearson_corr.abs()
    pearson_abs_tensor = torch.tensor(pearson_coor_abs.values, dtype=torch.float32)
    return pearson_abs_tensor

def create_sparse_edges_with_threshold(
    features_correlation: torch.Tensor, threshold: float
):
    n = int(features_correlation.shape[0])

    edges = []

    for i in range(n):
        for j in range(n):

            # skip identity column
            if i == j:
                continue

            if features_correlation[i, j] > threshold:
                edges.append((i, j))

    edge_index = torch.tensor(edges).T

    return edge_index

def create_sparse_edges_with_knn(features_correlation: torch.Tensor, k: int):
    n = int(features_correlation.shape[0])
    distance_matrix = 1 - features_correlation
    
    # used to avoid including self
    distance_matrix.fill_diagonal_(float('inf'))

    _, indices = torch.topk(distance_matrix, k=k, dim=1, largest=False)

    edges = []

    for i in range(n):
        for neighbor_idx in indices[i]:
            edges.append((i, neighbor_idx.item()))

            # undirected
            edges.append((neighbor_idx.item(), i))
    
    edge_index = torch.tensor(edges).T

    return edge_index