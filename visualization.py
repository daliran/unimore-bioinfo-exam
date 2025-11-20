import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

""" def show_correlations(features_correlation):
    df_temp = pearson_corr.abs()

    n_patients = df_temp.shape[0]
    cols = 10
    rows = int(np.ceil(n_patients / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20), sharex=True, sharey=True)
    ymin, ymax = -1, 1

    for i, ax in enumerate(axes.flat):
        if i < n_patients:
            ax.plot(df_temp.iloc[i, :].values, color='steelblue', linewidth=0.8)
            ax.set_title(f"P{i}", fontsize=8)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, linewidth=0.2)
        else:
            ax.axis('off')

    fig.suptitle("Pearson correlation di ciascun paziente con tutti gli altri", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show() """


# visualize(out, color=labels)
""" def graph_visualization(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show() """