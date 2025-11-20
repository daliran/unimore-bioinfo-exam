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
    sem = a.std(ddof=1) / np.sqrt(n)   # sample standard error
    # t critical
    t_crit = stats.t.ppf((1 + confidence) / 2., n-1)
    margin = sem * t_crit
    return mean, mean - margin, mean + margin

def main():
    seed = 42
    set_common_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_features, labels = load_raw_data("dataset_LUMINAL_A_B.csv")

    # TODO: check label balance

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

    # edge_index = create_sparse_edges_with_threshold(features_correlation, 0.4)

    edge_index = create_sparse_edges_with_knn(features_correlation, k=5)

    number_of_points = features.shape[0]

    accuracies = []

    for i in seed_pool:

        set_common_seeds(i)

        train_mask, val_mask, test_mask = create_dataset_masks(
            points=number_of_points,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            seed=seed,
        )

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
            input_channels=features.shape[1],
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
            eval_mask= test_mask,
            epochs=1000,
            patience=100,
            learning_rate=0.01,
            weight_decay=0.0002,
            verbose=False,
        )

        #print(f"Last train accuracy: {last_train_accuracy:.4f}")
        print(f"Test accuracy: {best_eval_accuracy:.4f}")

        accuracies.append(best_eval_accuracy)


    mean, lo, hi = mean_confidence_interval(accuracies)
    print(f"Mean={mean:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")
    

if __name__ == "__main__":
    main()
