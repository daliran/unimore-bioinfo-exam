import torch
from gnn_stats import seed_pool, set_common_seeds, mean_confidence_interval
from gnn_network import GNN, train_network
from gnn_dataset import (
    load_raw_data,
    remove_null_features,
    normalize_features,
    compute_abs_pearson_correlation,
    create_sparse_edges_with_threshold,
    create_sparse_edges_with_knn,
    create_dataset_masks,
    merge_dataset_masks,
)
from gnn_hpo import execute_hpo
from gnn_visualization import visualize_graph, show_correlations

def main():
    global_seed = 42
    set_common_seeds(global_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_features, labels = load_raw_data("dataset_LUMINAL_A_B.csv")

    # Check if labels are balanced
    #counts = torch.bincount(labels)
    #print(counts)

    cleanued_features = remove_null_features(raw_features)

    normalized_features = normalize_features(
        cleanued_features, target_dims=40, seed=global_seed
    )

    number_of_nodes = int(normalized_features.shape[0])
    number_of_dimensions = int(normalized_features.shape[1])

    features_correlation = compute_abs_pearson_correlation(normalized_features)

    #show_correlations(features_correlation)
    #edge_index = create_sparse_edges_with_threshold(features_correlation, 0.4)
    edge_index = create_sparse_edges_with_knn(features_correlation, k=5)

    visualize_graph(normalized_features, edge_index, number_of_nodes, labels)

    """ 
    train_mask, val_mask, test_mask = create_dataset_masks(
        points=number_of_points,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=global_seed,
    )

    study = execute_hpo(
        device = device,
        features=features,
        labels=labels,
        edge_index=edge_index,
        train_mask=train_mask,
        eval_mask=val_mask,
        trials=100,
    ) 
    """ 

    accuracies = []

    for seed in seed_pool:

        set_common_seeds(seed)

        train_mask, val_mask, test_mask = create_dataset_masks(
            points=number_of_nodes,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            seed=seed,
        )

        model = GNN(
            input_channels=number_of_dimensions,
            hidden_channels=32,
            output_channels=2,
            layers=4,
            dropout=0.25,
        )

        train_mask = merge_dataset_masks(train_mask, val_mask)

        train_accuracy, eval_accuracy, best_eval_accuracy = train_network(
            model=model,
            device=device,
            features=normalized_features,
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
