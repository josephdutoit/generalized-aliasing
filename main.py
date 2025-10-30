import os
from model import Model, ModelConfig
from data import GadDataset, synthetic_function
from vis import Plotter
import torch
import lightning as L

#TODO: Clean up directory structure so plots go in one place, models in another, etc.
#TODO: Make it so we can load config from file using command args and json
#TODO: Put lr and optimizer in config
MODEL_CONFIG = {
    "input_dim": 1,
    "hidden_dims": [64] * 4,
    "backbone_output_dim": 64,
    "small_num_basis_funcs": 4,
    "big_num_basis_funcs": 512,
    "output_dim": 1,
    "activation": "relu"
}

#TODO: Make this experiment code cleaner and modular
def __main__():
    config = ModelConfig(**MODEL_CONFIG)
    small_model = Model(config, model_type="small")

    if not os.path.exists("gad_dataset.pth"):
        big_dataset = GadDataset(
            num_samples=10000,
            input_range=(-5.0, 5.0),
            true_function=synthetic_function,
        )

        torch.save(big_dataset, "gad_dataset.pth")
    else:
        big_dataset = torch.load("gad_dataset.pth", weights_only=False)

    small_dataset = torch.utils.data.Subset(big_dataset, range(0, 50))

    loader = torch.utils.data.DataLoader(
        small_dataset,
        batch_size=32,
        shuffle=True,
    )

    trainer = L.Trainer(max_epochs=2000)
    trainer.fit(
        small_model,
        loader,
    )

    torch.save(small_model.model.state_dict(), config.small_model_save_path)

    plotter = Plotter()
    x = torch.linspace(-5, 5, 1000).unsqueeze(1)
    
    plotter.plot_function(
        x,
        synthetic_function,
        label="Synthetic Function",
    )
    plotter.plot_function(
        x,
        lambda x: small_model(x),
        label="Small Model Prediction",
    )
    plotter.save("function_plot.png")

    plotter.plot_function(
        x,
        lambda x: small_model.get_features(x),
        label="Basis Functions",
    )
    plotter.save("features_plot.png")

    
    loader = torch.utils.data.DataLoader(
        big_dataset,
        batch_size=128,
        shuffle=True,
    )


    big_model = Model(config, model_type="big")

    trainer = L.Trainer(max_epochs=2000)
    trainer.fit(
        big_model,
        loader,
    )

    torch.save(big_model, "big_model.pth")

    plotter = Plotter()
    x = torch.linspace(-5, 5, 1000).unsqueeze(1)

    plotter.plot_function(
        x,
        synthetic_function,
        label="Synthetic Function",
    )
    plotter.plot_function(
        x,
        lambda x: big_model(x),
        label="Big Model Prediction",
    )
    plotter.save("big_function_plot.png")

    plotter.plot_function(
        x,
        lambda x: big_model.get_features(x),
        label="Basis Functions",
    )
    plotter.save("big_features_plot.png")


if __name__ == "__main__":
    __main__()