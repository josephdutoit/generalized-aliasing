import torch
from matplotlib import pyplot as plt
import numpy as np

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

    def show(self):
        plt.legend()
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

            means = np.mean(vals, axis=0)
            mins = np.min(vals, axis=0)
            maxs = np.max(vals, axis=0)

            plt.fill_between(n_vals, mins, maxs, alpha=0.2)
            plt.plot(n_vals, means, label=f"Mean {key}")
            plt.xlabel("Number of Features (n)")
            plt.ylabel("Norm Value")
            plt.title(f"Norm: {key}")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
