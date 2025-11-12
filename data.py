import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class DataConfig:
    dataset_path: str = "gad_dataset.pth"
    num_samples_big: int = 10000
    num_samples_small: int = 50
    input_range: tuple[float, float] = (-5.0, 5.0)
    true_function: callable = None
    save: bool = True

#TODO: Make a dataset config class
#TODO: Make a class that uses a linear combination of some specifieed basis functions
def synthetic_function(X: torch.Tensor, N_terms: int = 50) -> torch.Tensor:
    n_values = torch.arange(1, N_terms + 1, dtype=X.dtype, device=X.device)
    arg = X.unsqueeze(-1) * n_values * n_values
    sin_terms = torch.sin(arg) / n_values
    return torch.sum(sin_terms, dim=-1)

class GadDataset(Dataset):
    def __init__(self, config: DataConfig):
        self.num_samples = config.num_samples_big
        self.input_range = config.input_range
        self.true_function = config.true_function or synthetic_function
        self.X = torch.rand(config.num_samples_big, 1) * (config.input_range[1] - config.input_range[0]) + config.input_range[0]
        self.y = config.true_function(self.X)
        
        if config.save:
            torch.save(self, config.dataset_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], synthetic_function(self.X[idx])

    def __to_tensor__(self):
        return self.X, self.y

    def __len__(self):
        return self.num_samples
    
if __name__ == "__main__":
    from viz import Plotter
    x = torch.linspace(-5, 5, 1000).unsqueeze(-1)
    plotter = Plotter()
    plotter.plot_function(x, lambda x: synthetic_function(x))
    plotter.save("synthetic_function.png")