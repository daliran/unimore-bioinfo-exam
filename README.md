# Task
The objective of this lab is to perform a graph node classification using a GNN.
In the dataset each node is a patient, the features are genes expressions and the labels represent two types of breast cancer: Luminal A/Luminal B.
The graph edges are not provided in the dataset and must be computed using Pearson correlation.

This lab will be performed using transductive learning.
In addition, the GNN performance needs to be compared with the performance of an MLP.

# Data preparation
The dataset contains 100 patients and around 1000 features.
In the provided datasets, the labels are balanced. 
We have 50% Luminal A and 50% Luminal B.

The gene expression data contains null/zero values across all patients.
These needs to be removed because they introduce noise in the learning process.

After the cleanup, the data needs to be scaled so that each feature has mean 0 and variance 1. This is called standard scaling.
This process is necessary to make the learning of the neural network easier and to avoid giving a preference to features with higher values over features with lower values.

Even after the cleanup, the number of features is still too high.
This is an issue, because we incur in the curse of dimensionality problem, which makes all points in the dataset too far away.

We need to apply dimensionality reduction.
A possible approach is to apply PCA.

Dimensionality reduction must be applied after the standard scaling.

Normally the transformations of both scaling and dimensionality reduction must be learned only from the training set, then applied to the validation and test set, to avoid leakage of statistics.
Since in this context we perform transductive learning, the "filtering" is done by masking the labels, while the GNN receives all data.
This requires to appply scaling and dimensionality reduction on all data, not only on the traning set.

# Graph edges calculation
The idea is to calculate the correlation of each patient using the Pearson correlation, assuming a linear correlation.
Other approaches can be followed using Spearman or Kendal correlation but they don't seem to provide any extra benefit over Pearson.

By comparing the correlation of each patient with all the other we can see that in most cases the values is low.
In some cases we have values going over 0.3/0.4 (excluding the diagonal, representing the correlation of each patient with itself).

Two possibilities are:
* Apply a threshold on the value of the correlation and create binary edges on the patients with a value above the threshold.
* Apply K-nn on and assign to each patient a binary edge on the closest k neighbors, using the value of the correlation.

In both cases the results is a sparse binary adjacency matrix of an undirected graph, expressed with an edges list.

# Model

The GNN can be created using the following types of layers:
* GCN
* GraphSage
* GAT

The choice for this lab is to use GCN.

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
TODO

# Test with MLP
TODO