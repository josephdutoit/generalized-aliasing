import torch
from torch.utils.data import Dataset

#TODO: Make a class that uses a linear combination of some specifieed basis functions
def synthetic_function(X: torch.Tensor, N_terms: int = 50) -> torch.Tensor:
    n_values = torch.arange(1, N_terms + 1, dtype=X.dtype, device=X.device)
    arg = X.unsqueeze(-1) * n_values
    sin_terms = torch.sin(arg) / n_values
    return torch.sum(sin_terms, dim=-1)

class GadDataset(Dataset):
    def __init__(
        self, 
        num_samples: int, 
        input_range: tuple[float, float],
        true_function: callable=synthetic_function,
    ):
        self.num_samples = num_samples
        self.X = torch.rand(num_samples, 1) * (input_range[1] - input_range[0]) + input_range[0]
        self.y = true_function(self.X)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], synthetic_function(self.X[idx])

    def __to_tensor__(self):
        return self.X, self.y

    def __len__(self):
        return self.num_samples