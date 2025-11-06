import torch
from matplotlib import pyplot as plt

class Plotter:
    def __init__(self):
        plt.figure(figsize=(10, 6))
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Function Plot")
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