import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm, colors

class Plotter:
    def __init__(self, figsize=(10, 6)):
        plt.figure(figsize=figsize)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)

    def plot_function(
        self,
        x: torch.Tensor,
        f: callable,
        label: str = "Function",
    ):
        with torch.no_grad():
            y = f(x).cpu().numpy()
        x = x.cpu().numpy()
        plt.plot(x, y, label=label)

    def plot_min_max_fill_between(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str = None,
    ):
        means = np.mean(y, axis=0)
        mins = np.min(y, axis=0)
        maxs = np.max(y, axis=0)
        plt.plot(x, means, label=label)
        plt.fill_between(x, mins, maxs, alpha=0.2)

    def show(self):
        plt.show()

    def save(self, filename: str):
        plt.legend()
        plt.savefig(filename)
        plt.clf()

    def plot_data_points(self, x, y, label: str = "Data Points", color: str = 'red'):
        plt.scatter(x, y, color=color, label=label)

    def plot_eval(
        self,
        norms_dict: dict,
        n_vals: list[int],
    ):
        n_items = len(norms_dict)
        for i, (key, vals) in enumerate(norms_dict.items()):
            plt.subplot(n_items, 1,  i + 1)

            self.plot_min_max_fill_between(
                x=n_vals,
                y=vals,
                label=f"Norm: {key}",
            )
            plt.xlabel("Number of Features (n)")
            plt.ylabel("Norm Value")
            plt.title(f"Norm: {key}")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()

    def plot_svals(
        self,
        svd_dict: dict,
    ):
        # assume svd_dict maps n (int) -> array-like of shape (runs, n_svals)
        n_keys = sorted(svd_dict.keys())
        if len(n_keys) == 0:
            return

        cmap = cm.get_cmap("viridis")
        norm = colors.Normalize(vmin=0, vmax=max(1, len(n_keys) - 1))

        for i, n in enumerate(n_keys):
            vals = np.array(svd_dict[n])
            mean_svals = np.mean(vals, axis=0)
            x = np.arange(1, mean_svals.shape[0] + 1)
            color = cmap(norm(i))
            plt.plot(x, mean_svals, color=color, label=f"n={n}")

        # colorbar that maps line colors to n values
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # Explicitly associate the colorbar with the current axes
        ax = plt.gca()  # Get the current axes
        cbar = plt.colorbar(sm, ax=ax, ticks=np.linspace(0, max(n_keys), num=len(n_keys)))
        cbar.set_label("n (number of features)")

        # Nicer tick labels showing actual n values
        if len(n_keys) <= 10:
            cbar.set_ticks(range(len(n_keys)))
            cbar.set_ticklabels([str(k) for k in n_keys])

        plt.xlabel("Singular value index")
        plt.ylabel("Mean singular value")
        plt.title("Mean singular values across runs")
        plt.grid(True)
        plt.tight_layout()
