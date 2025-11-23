import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(
        self, input_channels, hidden_channels, output_channels, num_layers, dropout
    ):
        super().__init__()

        self.dropout = dropout
        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(input_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))

        self.layers.append(torch.nn.Linear(hidden_channels, output_channels))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # activation and dropout on all layers except the last
            if i < len(self.layers) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


def calculate_mlp_accuracy(out, labels, mask):
    prediction = out.argmax(dim=1)
    correct_labels = prediction[mask] == labels[mask]
    accuracy = int(correct_labels.sum()) / int(mask.sum())
    return accuracy


def train_mlp_network(
    model: MLP,
    device,
    features,
    labels,
    train_mask,
    eval_mask,
    epochs,
    patience,
    learning_rate,
    weight_decay,
    verbose=False,
):
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    eval_mask = eval_mask.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_eval_accuracy = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            out = model(features)
            train_accuracy = calculate_mlp_accuracy(out, labels, train_mask)
            eval_accuracy = calculate_mlp_accuracy(out, labels, eval_mask)

        if verbose:
            print(
                f"Epoch {epoch:03d} | Loss={loss:.4f} | Train={train_accuracy:.4f} | Eval={eval_accuracy:.4f}"
            )

        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch:03d}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_accuracy, eval_accuracy, best_eval_accuracy
