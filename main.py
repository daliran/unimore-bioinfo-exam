import torch
from seeds import set_common_seeds, seed_pool
from network import GNN, train_network
from dataset import (
    load_raw_data,
    normalize_and_cleanup_features,
    compute_abs_pearson_correlation,
    create_sparse_edges_with_threshold,
    create_sparse_edges_with_knn,
    create_dataset_masks,
    merge_dataset_masks,
)
from hpo import execute_hpo

import numpy as np
from scipy import stats


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    mean = a.mean()
    sem = a.std(ddof=1) / np.sqrt(n)  # sample standard error
    # t critical
    t_crit = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    margin = sem * t_crit
    return mean, mean - margin, mean + margin


def main():
    global_seed = 42
    set_common_seeds(global_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_features, labels = load_raw_data("dataset_LUMINAL_A_B.csv")

    # TODO: check label balance

    # features = normalize_and_cleanup_features(raw_features, target_dims=40, seed=global_seed)
    # features_correlation = compute_abs_pearson_correlation(features)
    # edge_index = create_sparse_edges_with_threshold(features_correlation, 0.4)
    # edge_index = create_sparse_edges_with_knn(features_correlation, k=5)

    raw_features = raw_features[:, (raw_features != 0).any(dim=0)]
    number_of_points = raw_features.shape[0]

    accuracies = []

    for seed in seed_pool:

        set_common_seeds(seed)

        train_mask, val_mask, test_mask = create_dataset_masks(
            points=number_of_points,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            seed=seed,
        )

        features = normalize_and_cleanup_features(
            raw_features, target_dims=40, seed=global_seed
        )

        raw_features_numpy = raw_features.numpy()
        scaler = StandardScaler()
        train_scaled_features = scaler.fit_transform(raw_features_numpy[train_mask])
        test_scaled_features = scaler.transform(raw_features_numpy[test_mask])

        pca = PCA(n_components=40, random_state=seed)
        train_reduced_features_np = pca.fit_transform(train_scaled_features)
        test_reduced_features_np = pca.transform(test_scaled_features)

        train_reduced = torch.tensor(train_reduced_features_np, dtype=torch.float32)
        test_reduced = torch.tensor(test_reduced_features_np,  dtype=torch.float32)

        features_correlation = compute_abs_pearson_correlation(train_reduced)
        edge_index = create_sparse_edges_with_knn(features_correlation, k=5)

        """ study = execute_hpo(
            device = device,
            features=features,
            labels=labels,
            edge_index=edge_index,
            train_mask=train_mask,
            eval_mask=val_mask,
            trials=100,
        ) """

        model = GNN(
            input_channels=train_reduced.shape[1],
            hidden_channels=32,
            output_channels=2,
            layers=4,
            dropout=0.25,
        )

        train_mask = merge_dataset_masks(train_mask, val_mask)

        last_train_accuracy, last_eval_accuracy, best_eval_accuracy = train_network(
            model=model,
            device=device,
            features=features,
            labels=labels,
            edge_index=edge_index,
            train_mask=train_mask,
            eval_mask=test_mask,
            epochs=1000,
            patience=100,
            learning_rate=0.01,
            weight_decay=0.0002,
            verbose=False,
        )

        # print(f"Last train accuracy: {last_train_accuracy:.4f}")
        print(f"Test accuracy: {best_eval_accuracy:.4f}")

        accuracies.append(best_eval_accuracy)

    mean, lo, hi = mean_confidence_interval(accuracies)
    print(f"Mean={mean:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")


if __name__ == "__main__":
    main()
