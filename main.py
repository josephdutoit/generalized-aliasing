import os
from model import ModelConfig
from data import DataConfig, synthetic_function
from experiments import Experiment, ExperimentConfig
import click

#TODO: Clean up directory structure so plots go in one place, models in another, etc.
#TODO: Make it so we can load config from file using command args and json
#TODO: Put lr and optimizer in config


@click.command()
@click.option('--num_repeats', default=1, help='Number of experiment repeats')
@click.option('--samplewise', is_flag=True, help='Use samplewise iteration')
def main(num_repeats, samplewise=False):
    
    experiment_config = ExperimentConfig(
        num_repeats=5,
        samplewise=False,
        n_vals=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        # n_vals=[5, 10],
        experiment_name="interp_50_normalized",
        save_dir="results/",
        accelerator="gpu",
    )

    data_config = DataConfig(
        dataset_path="gad_dataset2.pth",
        num_samples_big=10000,
        num_samples_small=50,
        input_range=(-5.0, 5.0),
        true_function=synthetic_function,
        save=True,
    )

    model_config = ModelConfig(
        input_dim=1,
        small_num_basis_funcs=15, # TODO: Add logic so that we don't have to set this if modelwise descent
        big_num_basis_funcs=512,
        backbone_output_dim=64,
        hidden_dims=[64] * 4,
        output_dim=1,
        activation="relu",
    )

    experiment = Experiment(experiment_config)
    experiment.run(
        model_config=model_config, 
        data_config=data_config,
        parallel_k=4,
    )
    

if __name__ == "__main__":
    main()