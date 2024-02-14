import matplotlib.pyplot as plt
import numpy as np

def plot_embeddings(embeddings, labels, titles, cmap='tab10', figsize=(13, 21), ncols=3, save_as=None, dpi=300):
    assert len(embeddings) == len(labels) == len(titles), "embeddings, labels and titles lists must have the same length"

    plt.rcParams['font.size'] = 18
    
    rows = int(np.ceil(len(embeddings) / ncols))
    fig, axs = plt.subplots(rows, ncols, figsize=figsize)
    axs = axs.flatten()  # Flatten the array of axes to easily iterate over it
    
    for i, (data, label, title) in enumerate(zip(embeddings, labels, titles)):
        scatter = axs[i].scatter(data[:, 0], data[:, 1], c=label[:data.shape[0]], cmap=cmap)
        axs[i].set_title(title)
        axs[i].set_xlabel('Component 1', fontsize=18)
        axs[i].set_ylabel('Component 2', fontsize=18)
        cbar = fig.colorbar(scatter, ax=axs[i])
        cbar.set_label('labels', fontsize=18)
    
    # If there are less embeddings than subplots, remove the extra subplots
    if len(embeddings) < len(axs):
        for ax in axs[len(embeddings):]:
            fig.delaxes(ax)
    
    if save_as:
        plt.savefig(save_as, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()