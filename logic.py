import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv
import numpy as np
import random

class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_gnn_epoch(
    model, optimizer, criterion, features, labels, edge_index, train_mask
):
    optimizer.zero_grad()
    out = model(features, edge_index)
    loss = criterion(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss


def train_gnn(
    model: GCN, epochs: int, features, labels, edge_index, train_mask, verbose=False
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss = train_gnn_epoch(
            model, optimizer, criterion, features, labels, edge_index, train_mask
        )

        if verbose:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")


def test_gnn(model: GCN, features, labels, edge_index, test_mask):
    model.eval()
    out = model(features, edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[test_mask] == labels[test_mask]
    test_acc = int(test_correct.sum()) / int(test_mask.sum())
    return test_acc


def data_preparation(file_path: str, number_of_reduced_dimensions: int):
    df = pd.read_csv(file_path)

    # series
    label_column = df.iloc[:, 0]

    # data frame
    feature_df = df.iloc[:, 1:]

    # index
    zero_expresison_genes_columns = feature_df.columns[(feature_df == 0).all()]

    # data frame
    feature_cleaned = feature_df.drop(columns=zero_expresison_genes_columns)

    scaler = StandardScaler()

    # np array
    features_scaled = scaler.fit_transform(feature_cleaned.values)

    pca = PCA(n_components=number_of_reduced_dimensions)

    # np array
    features_pca = pca.fit_transform(features_scaled)

    # tensors
    features = torch.tensor(features_pca, dtype=torch.float32)
    labels = torch.tensor(label_column == "Luminal B    ", dtype=torch.long)

    return features, labels


def create_sparse_edges_with_threshold(
    data_points_correlation: torch.Tensor, threshold: int
):

    n = int(data_points_correlation.shape[0])

    edges = []

    for i in range(n):
        for j in range(n):

            # skip identity column
            if i == j:
                continue

            if data_points_correlation[i, j] > threshold:
                edges.append((i, j))

    edge_index = torch.tensor(edges).T

    return edge_index

def prepare_dataset_masks(
    data_points_number: int, train_split: int = 0.8, validation_split = 0.1, test_split: int = 0.1):

    num_train = int(train_split * data_points_number)
    num_validation = int(validation_split * data_points_number)
    num_test = int(test_split * data_points_number)

    perm = torch.randperm(data_points_number)

    train_idx = perm[:num_train]
    test_idx = perm[num_test:]

    train_mask = torch.zeros(data_points_number, dtype=torch.bool)
    test_mask = torch.zeros(data_points_number, dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return train_mask, test_mask

def show_correlation():
    pass

def show_graph():
    pass

##-------------------------------------------

def load_raw_data(file_path):
    df = pd.read_csv(file_path)

    label_column = df.iloc[:, 0]
    raw_labels = torch.tensor(label_column == "Luminal B    ", dtype=torch.long)

    feature_df = df.iloc[:, 1:]
    raw_features = torch.tensor(feature_df.values, dtype=torch.float32)

    return raw_features, raw_labels

def normalize_and_cleanup_features(raw_features, target_dims):

    # remove features with 0 values
    mask = (raw_features != 0).any(dim=0)

    filtered_features = raw_features[:, mask]

    # apply standard scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(filtered_features)

    # apply PCA for dimensionality reduction
    pca = PCA(n_components=target_dims)
    reduced_features = pca.fit_transform(scaled_features)

    return torch.tensor(reduced_features, dtype=torch.float32)

def compute_abs_pearson_correlation(features: torch.Tensor):
    df_temp = pd.DataFrame(features)
    pearson_corr = df_temp.T.corr(method="pearson")
    pearson_coor_abs = pearson_corr.abs()
    pearson_abs_tensor = torch.tensor(pearson_coor_abs.values, dtype=torch.float32)
    return pearson_abs_tensor

def execute_experiment(features, labels):
    # dataset split only training+validation and test
    # training
    # testing
    pass

def set_all_seeds(seed=42):

    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    set_all_seeds(42)

    raw_features, labels = load_raw_data("dataset_LUMINAL_A_B.csv")
    features = normalize_and_cleanup_features(raw_features, target_dims=40)
    features_correlation = compute_abs_pearson_correlation(features)

    # TODO: dataset split training, validation, test and hyper parameter selection

    #seeds = [1234567]
    #for seed in seeds:
    #    set_all_seeds(seed)
    #    execute_experiment(raw_features, labels)
    

    features2, labels2 = data_preparation("dataset_LUMINAL_A_B.csv", 40)
    correlation = compute_abs_pearson_correlation(features2)
    pass

    '''
    edges = create_sparse_edges_with_threshold(correlation, 0.4)

    number_of_points = features.shape[0]
    train_mask, test_mask = prepare_dataset_masks(number_of_points)
    
    model = GCN(input_channels=features.shape[1], hidden_channels=8, output_channels=2)

    train_gnn(model, 100, features, labels, edges, train_mask, verbose=False)
    test_accuracy = test_gnn(model, features, labels, edges, test_mask)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    '''

if __name__ == "__main__":
    main()