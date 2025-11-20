import optuna
from optuna.trial import Trial
from network import GNN, train_network, evaluate_network

def objective(trial: Trial, device, features, labels, edge_index, train_mask, eval_mask):

    hidden_channels = trial.suggest_categorical("hidden_channels", [8, 16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.7)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    epochs = 200
    input_channels = features.size(1)

    model = GNN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_channels=2,
        layers=num_layers,
        dropout=dropout,
    )

    train_network(
        model=model,
        device=device,
        features=features,
        labels=labels,
        edge_index=edge_index,
        train_mask=train_mask,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        verbose=False,
    )

    eval_accuracy = evaluate_network(
        model=model,
        device=device,
        features=features,
        labels=labels,
        edge_index=edge_index,
        eval_mask=eval_mask,
    )

    return eval_accuracy


def execute_hpo(trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    pass
