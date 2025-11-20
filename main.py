import torch
from seeds import set_common_seeds
from network import GNN, train_network, evaluate_network
from dataset import (
    load_raw_data,
    normalize_and_cleanup_features,
    compute_abs_pearson_correlation,
    create_sparse_edges_with_threshold,
    create_sparse_edges_with_knn,
    create_dataset_masks,
    merge_dataset_masks
)

def main():
    seed = 42
    set_common_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_features, labels = load_raw_data("dataset_LUMINAL_A_B.csv")

    #TODO: check label balance

    features = normalize_and_cleanup_features(raw_features, target_dims=40, seed=seed)
    features_correlation = compute_abs_pearson_correlation(features)

    # TODO: dataset split training, validation, test and hyper parameter selection
    # first isolate the test set, then create the training and validation masks
    # execute_hpo()

    # seeds = [1234567]

    # for seed in seeds:
    #    set_common_seeds(seed)
    #    execute_experiment(raw_features, labels)
    # dataset split only training+validation and test
    # training
    # testing

    #edge_index = create_sparse_edges_with_threshold(features_correlation, 0.4)

    edge_index = create_sparse_edges_with_knn(features_correlation, k = 5)

    number_of_points = features.shape[0]

    train_mask, val_mask, test_mask = create_dataset_masks(
        points=number_of_points,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=seed,
    )

    test_mask = merge_dataset_masks(val_mask, test_mask)

    model = GNN(
        input_channels=features.shape[1],
        hidden_channels=8,
        output_channels=2,
        layers=2,
        dropout=0.5,
    )

    last_train_accuracy = train_network(
        model=model,
        device=device,
        features=features,
        labels=labels,
        edge_index=edge_index,
        train_mask=train_mask,
        epochs=200,
        learning_rate=0.01,
        weight_decay=5e-4,
        verbose=False,
    )

    test_accuracy = evaluate_network(
        model=model,
        device=device,
        features=features,
        labels=labels,
        edge_index=edge_index,
        eval_mask=test_mask,
    )

    print(f"Last train accuracy: {last_train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    # according to gpt 0.6-0.75 is ok, 0.8-0.85 excellent, more is unrealistic

if __name__ == "__main__":
    main()
