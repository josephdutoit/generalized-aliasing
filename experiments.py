import os
import torch
import lightning as L
from model import Model, ModelConfig
from dataclasses import dataclass
from data import DataConfig, GadDataset
from viz import Plotter
from copy import deepcopy
import concurrent.futures
from typing import Optional, Sequence


#TODO: Can we find someway that passes stuff rather than saving and loading?

@dataclass
class ExperimentConfig:
    num_repeats: int = 1
    samplewise: bool = False
    n_vals: list[int] = None
    experiment_name: str = "experiment"
    save_dir: str = "results/"
    accelerator: str = "auto"
    verbose: bool = False

@dataclass
class RunConfig:
    input_range: tuple[float, float] = (-5.0, 5.0)
    small_model_epochs: int = 2000
    big_model_epochs: int = 5000
    small_model_batch_size: int = 32
    big_model_batch_size: int = 128
    function_plot_path: str = "function_plot.png"
    features_plot_path: str = "features_plot.png"
    save_path: str = "results/experiment"
    run_name: str = "run"
    plot: str = False
    accelerator: str = "auto"

class Run:
    def __init__(self, config: RunConfig):
        self.config = config

    # TODO: Refactor this so it is nicer and cleaner
    def run(self, model_config: ModelConfig, data_config: DataConfig):
        
        # Make a verbose something for printing
        print(f"Starting run: {self.config.run_name}")
        
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
        else:
            raise FileExistsError(f"Run directory {self.config.save_path} already exists.")


        model_config.small_model_save_path = os.path.join(
            self.config.save_path,
            model_config.small_model_save_path
        )

        model_config.big_model_save_path = os.path.join(
            self.config.save_path,
            model_config.big_model_save_path
        )

        small_model = Model(model_config, model_type="small")
        big_dataset = torch.load(data_config.dataset_path, weights_only=False)

        small_dataset = torch.utils.data.Subset(big_dataset, range(0, data_config.num_samples_small))
        loader = torch.utils.data.DataLoader(
            small_dataset,
            batch_size=self.config.small_model_batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            big_dataset,
            batch_size=1,
            shuffle=False,
        )
        trainer = L.Trainer(
            max_epochs=self.config.small_model_epochs,
            accelerator=self.config.accelerator,
            enable_progress_bar=False,
        )
        trainer.fit(
            small_model,
            loader,
        )

        trainer.test(
            small_model,
            test_loader,
        )

        torch.save(
            small_model.model.state_dict(), 
            model_config.small_model_save_path
        )

        big_model = Model(model_config, model_type="big")
        loader = torch.utils.data.DataLoader(
            big_dataset,
            batch_size=self.config.big_model_batch_size,
            shuffle=True,
        )
        trainer = L.Trainer(
            max_epochs=self.config.big_model_epochs,
            accelerator=self.config.accelerator,
            enable_progress_bar=False,
        )

        trainer.fit(
            big_model,
            loader,
        )

        torch.save(
            big_model.model.state_dict(), 
            model_config.big_model_save_path
        )

        big_model.model.eval()

        gad = big_model.model.compute_gad(
            big_dataset.X, 
            data_config.num_samples_small,
            interval=data_config.input_range,
            normalize_features=True,
        )
        gad["small_model_val_loss"] = small_model.final_val_loss
        torch.save(gad, os.path.join(self.config.save_path, "gad_results.pth")) #Fix this pathing

        # TODO: Move plotting to separate function
        if self.config.plot:
            plotter = Plotter()
            x = torch.linspace(
                self.config.input_range[0], 
                self.config.input_range[1], 
                1000
            ).unsqueeze(1)
            
            plotter.plot_function(
                x,
                data_config.true_function,
                label="Synthetic Function",
            )
            plotter.plot_function(
                x,

                lambda x: small_model(x),
                label="Small Model Prediction",
            )
            
            plotter.save(os.path.join(self.config.save_path, "small_" + self.config.function_plot_path))

            plotter.plot_function(
                x,
                lambda x: small_model.get_features(x),
                label="Basis Functions",
            )
            plotter.save(os.path.join(self.config.save_path, "small_" + self.config.features_plot_path))

            plotter.plot_function(
                x,
                data_config.true_function,
                label="Synthetic Function",
            )
            plotter.plot_function(
                x,
                lambda x: big_model(x),
                label="Big Model Prediction",
            )
            plotter.save(os.path.join(self.config.save_path, "big_" + self.config.function_plot_path))

            plotter.plot_function(
                x,
                lambda x: big_model.get_features(x, normalize=True),
                label="Basis Functions",
            )
            plotter.save(os.path.join(self.config.save_path, "big_" + self.config.features_plot_path))


def _run_worker(args):
    """
    Worker executed in a separate process. args is a tuple:
      (run_model_config, run_data_config, run_config, gpu)
    """
    run_model_config, run_data_config, run_config, gpu = args
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    run = Run(run_config)
    run.run(run_model_config, run_data_config)
    return run_config.run_name

# TODO: Parallelize all the runs
class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, model_config: ModelConfig, data_config: DataConfig, parallel_k: int = 1, gpu_devices: Optional[Sequence[int]] = None):
        if not os.path.exists(os.path.join(self.config.save_dir, self.config.experiment_name)):
            os.makedirs(os.path.join(self.config.save_dir, self.config.experiment_name))
        else:
            raise FileExistsError(f"Experiment directory {self.config.experiment_name} already exists.")
        
        if not os.path.exists(data_config.dataset_path):
            dataset = GadDataset(data_config)
            torch.save(dataset, data_config.dataset_path)

        # build task list
        tasks = []
        for repeat in range(self.config.num_repeats):
            torch.random.manual_seed(repeat + 42)
            for n_val in self.config.n_vals:
                run_name = f"{self.config.experiment_name}_run_{repeat+1}_n_{n_val}"
                run_model_config = deepcopy(model_config)
                run_data_config = deepcopy(data_config)
                if not self.config.samplewise:
                    run_model_config.small_num_basis_funcs = n_val

                run_cfg = RunConfig(
                    run_name=run_name,
                    plot=True,
                    save_path=os.path.join(self.config.save_dir, self.config.experiment_name, run_name)
                )

                if self.config.samplewise:
                    run_data_config.num_samples_small = n_val

                tasks.append((run_model_config, run_data_config, run_cfg))

        if parallel_k is None or parallel_k <= 1:
            for run_model_config, run_data_config, run_cfg in tasks:
                run = Run(run_cfg)
                run.run(run_model_config, run_data_config)
        else:
            max_workers = min(parallel_k, len(tasks))
            use_gpus = list(gpu_devices) if gpu_devices else None
            serialized_tasks = []
            for i, (rmc, rdc, rc) in enumerate(tasks):
                assigned_gpu = None
                if use_gpus:
                    assigned_gpu = use_gpus[i % len(use_gpus)]
                serialized_tasks.append((rmc, rdc, rc, assigned_gpu))

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_run_worker, t) for t in serialized_tasks]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        name = fut.result()
                        print(f"Finished run: {name}")
                    except Exception as e:
                        print(f"Run failed: {e}")