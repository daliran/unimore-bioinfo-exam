import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx


def show_correlations(features_correlation, title="Absolute Pearson correlations"):
    corr = features_correlation.detach().cpu().abs()

    n_patients = corr.shape[0]
    cols = 10
    rows = int(np.ceil(n_patients / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20), sharex=True, sharey=True)
    ymin, ymax = -1, 1

    axes = np.ravel(axes)

    for i, ax in enumerate(axes):
        if i < n_patients:
            row_vals = corr[i].numpy()
            ax.plot(row_vals, color="steelblue", linewidth=0.8)
            ax.set_title(f"P{i}", fontsize=8)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, linewidth=0.2)
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def visualize_graph(
    features, edge_index, num_nodes, labels, title="Graph visualization"
):
    data = Data(features=features, edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, k=1.0, seed=42)
    y = labels.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))

    cmap = plt.cm.get_cmap("tab10", np.unique(y).size)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=y, cmap=cmap, node_size=100)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    cbar = plt.colorbar(nodes, ticks=np.unique(y))
    cbar.set_label("Node label")
    plt.title(title)
    plt.axis("off")
    plt.show()
