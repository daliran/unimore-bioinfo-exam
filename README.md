# Task

The task executed in this lab is to perform a graph node classification using a GNN.
Each node represents a patient and the features are gene expressions.
Each node can be classifed as Luminal A or Luminal B (breast cancer).

The graph edges are not provided and must be computed using Pearson correlation.

Transductive learning -> predict node labels of unlabeled nodes in the exising graph.

Inductive learning -> predict node labels using a completely different graph.

In this lab we use transductive learning.

As and extra, the lab rquires to compare the results of the GNN with the results of a MLP classifier.

# Data preparation

GCN
GraphSage
GAT

Pearson
Spearman
Kendal

Normally standard scaling and PCA should be fit only to the training data to avoid leakage of statistics, then validation and test should be transformed using the learned transformation.

In transductive learning all the data is provided to the GNN, so we cannot fit only the training data otherwise the features will have incompatible values.
The fit must be done on the entire dataset. This is a special case.

Labels in the dataset are balanced 50-50.

# HPO phase
Split dataset into training, validation, test using a fixed global seed.
HPO is performed using the validation set in eval -> HPO must not see the test split

The steps are:
* perform a single optuna study, running all trials with a fixed common seed and find the trial with the best hyperparameters using the test accuracy as metric.
* train the real network multiple times, with different seeds, using the best hyperparameter.
* calculate the CI of the test accuracy across all trainings.
* check the variance of the CI.
    * if the variance is small the result of the HPO can be trusted.
    * if the variance is high the result of HPO cannot be trusted.

If the CI has a low variance, then the test accuracy of the HPO should not be too different than the real test accuracy.

If the variance is high it means that the hyper parameter are sensitive to random initialization, so we cannot use a fixed common seed across trials.
In this situation it is possible to get a trial with a very high test accuracy due to a lucky seed, but all other trials may have a significally lower accuracy.

The tested hyperparametes are:
* Learning rate
* 

# Confidence interval (CI)