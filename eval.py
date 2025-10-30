import torch

small_model = torch.load("small_model.pth", weights_only=False)
big_model = torch.load("big_model.pth", weights_only=False)

gad = big_model._compute_gad(torch.linspace(-5, 5, 1000).unsqueeze(1))