import os
import torch
import numpy as np
from experiments import ExperimentConfig
from viz import Plotter


class Evaluator(ExperimentConfig):
    def __init__(self, 
            config_path: str | None = None,
            experiment_config: ExperimentConfig | None = None,
    ):
        super().__init__(config_path)

        if experiment_config is not None:
            self.experiment_config = experiment_config
        elif config_path is not None:
            self.experiment_config = ExperimentConfig(config_path)
        else:
            raise ValueError("Either config_path or experiment_config must be provided.")

        self.results = None

    def evaluate(
        self, 
        keys: list[str] = ['m_tm_pinv', 'm_tu', 'p_n', 'theta_m', 'theta_u', 'm_tu_theta_u', 'small_model_val_loss']
    ):
        
        if self.results is not None:
            return self.results
        
        print(f"Evaluating experiment: {self.experiment_config.experiment_name}")
        
        # repeats = self.experiment_config.num_repeats
        # We did not finish run 3
        repeats = 5
        n_vals = self.experiment_config.n_vals

        norms_dict = {key: np.zeros((repeats-1, len(n_vals))) for key in keys}
        for repeat in range(repeats):
            if repeat == 0:
                continue
            for i, n in enumerate(n_vals):

                run = f"{self.experiment_config.experiment_name}_run_{repeat+1}_n_{n}"

                gad_data = torch.load(
                    os.path.join(
                        self.experiment_config.save_dir,
                        self.experiment_config.experiment_name,
                        run,
                        "gad_results.pth"
                    ),
                    map_location=torch.device('cpu')
                )

                m_tm = gad_data['m_tm']
                m_tm_pinv = torch.linalg.pinv(m_tm)

                m_tu_theta_u = gad_data['m_tu'] @ gad_data['theta_u']

                gad_data['m_tm_pinv'] = m_tm_pinv
                gad_data['m_tu_theta_u'] = m_tu_theta_u

                for key in keys:
                    val = torch.norm(gad_data[key], p=2).item()
                    norms_dict[key][repeat-1, i] = val

        self.results = norms_dict
        return self.results

    
    def plot_results(
        self,
        save_path: str | None = None,
    ):
        plotter = Plotter(figsize=((10, 6*len(self.results))))
        plotter.plot_eval(
            norms_dict=self.results,
            n_vals=self.experiment_config.n_vals,
        )
        if save_path is not None:
            plotter.save(save_path)


if __name__ == "__main__":
    experiment_config = ExperimentConfig(
        num_repeats=3,
        samplewise=False,
        n_vals=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        # n_vals=[5, 10],
        experiment_name="nn_interp_50_v2",
        save_dir="results/",
        accelerator="cpu",
    )
    evaluator = Evaluator(experiment_config=experiment_config)
    evaluator.evaluate()
    evaluator.plot_results(save_path=experiment_config.save_dir + experiment_config.experiment_name + "/evaluation_plot.png")