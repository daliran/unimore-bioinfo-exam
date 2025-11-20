import optuna
from optuna.trial import Trial
from functools import partial
from network import GNN, train_network


def objective(
    trial: Trial, device, features, labels, edge_index, train_mask, eval_mask
):

    hidden_channels = trial.suggest_categorical("hidden_channels", [8, 16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.7)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    epochs = 1000
    patience = 100
    input_channels = features.size(1)

    model = GNN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_channels=2,
        layers=num_layers,
        dropout=dropout,
    )

    last_train_accuracy, last_eval_accuracy, best_eval_accuracy = train_network(
        model=model,
        device=device,
        features=features,
        labels=labels,
        edge_index=edge_index,
        train_mask=train_mask,
        eval_mask = eval_mask,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        verbose=False,
    )

    return best_eval_accuracy


def execute_hpo(
    device, features, labels, edge_index, train_mask, eval_mask, trials=100
):

    wrapped_objective = partial(
        objective,
        device=device,
        features=features,
        labels=labels,
        edge_index=edge_index,
        train_mask=train_mask,
        eval_mask=eval_mask,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(wrapped_objective, n_trials=trials)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study
