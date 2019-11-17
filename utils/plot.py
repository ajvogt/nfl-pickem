import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_results(results, results_map):
    max_value = int(results_map.max())
    
    # Setting up colormap
    if results_map.min() < 0:
        colors = [(252/256, 79/256, 48/256),
                (1, 1, 1)]
    else:
        colors = [(1, 1, 1)]
    for i in range(max_value-1):
        colors.append((
            (i*(109 - 229)/max_value + 229)/256,
            (i*(114 - 174)/max_value + 174)/256,
            (i*(79 - 56)/max_value + 56)/256
        ))
    colors.append((109/256, 114/256, 79/256))
    my_cm = ListedColormap(colors)

    # plotting table results
    fig, ax = plt.subplots(figsize=(15, 7.5))
    cax = ax.matshow(results_map, 
                     cmap=my_cm, aspect=0.15, alpha=0.5)
    ax.set_xticklabels(['']+list(results.columns))
    ax.set_yticklabels([])
    ax.grid(False)
    cbar = fig.colorbar(cax, ticks=range(-1, max_value+1))
    labels = ['Already Picked', 'Not Picked']
    labels += ['Forecast %i weeks'%x for x in range(max_value)]
    cbar.ax.set_yticklabels(labels)
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            ax.text(x=j, y=i,
                            s=results.iloc[i, j],
                            va='center', ha='center',
                            fontsize=9)
    plt.show()

if __name__ == "__main__":
    print(__name__)
