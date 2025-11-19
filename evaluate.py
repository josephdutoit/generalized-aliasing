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
        self.svd_vals = None

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
        svd_vals = {n: [] for n in n_vals}
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
                m_tm_svd_vals = torch.linalg.svdvals(m_tm)

                # TODO: Find the fastest way to store and load these store as np arrays?
                svd_vals[n].append(m_tm_svd_vals.detach().cpu().numpy())

                m_tu_theta_u = gad_data['m_tu'] @ gad_data['theta_u']

                gad_data['m_tm_pinv'] = m_tm_pinv
                gad_data['m_tu_theta_u'] = m_tu_theta_u

                for key in keys:
                    val = torch.norm(gad_data[key], p=2).item()
                    norms_dict[key][repeat-1, i] = val

        self.norms_dict = norms_dict
        self.svd_vals = svd_vals
        return self.results

    
    def plot_norms(
        self,
        save_path: str | None = None,
    ):
        plotter = Plotter(figsize=((10, 6*len(self.norms_dict))))
        plotter.plot_eval(
            norms_dict=self.norms_dict,
            n_vals=self.experiment_config.n_vals,
        )
        if save_path is not None:
            plotter.save(save_path)

    def plot_individual_svd(
        self,
        save_dir: str,
    ):
        os.makedirs(save_dir, exist_ok=True)
        plotter = Plotter(figsize=(10, 6))
        for n in self.experiment_config.n_vals:
            save_path = os.path.join(
                save_dir,
                f"{self.experiment_config.experiment_name}_svd_n_{n}.png"
            )
            plotter.plot_min_max_fill_between(
                x=np.arange(1, len(self.svd_vals[n][0]) + 1),
                y=self.svd_vals[n],
                label=f"SVDs for n={n}",
            )

            plotter.save(save_path)

    def plot_all_svd(
        self,
        save_path: str,
    ):
        plotter = Plotter(figsize=(10, 6))
        plotter.plot_svals(svd_dict=self.svd_vals)
        plotter.save(save_path)



if __name__ == "__main__":
    experiment_config = ExperimentConfig(
        num_repeats=5,
        samplewise=False,
        n_vals=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        # n_vals=[5, 10],
        experiment_name="interp_50_normalized",
        save_dir="results/",
        accelerator="cpu",
    )
    evaluator = Evaluator(experiment_config=experiment_config)
    evaluator.evaluate()
    evaluator.plot_norms(save_path=experiment_config.save_dir + experiment_config.experiment_name + "/evaluation_plot.png")
    evaluator.plot_individual_svd(save_dir=experiment_config.save_dir + experiment_config.experiment_name + "/svd_plots/")
    evaluator.plot_all_svd(save_path=experiment_config.save_dir + experiment_config.experiment_name + "/all_svd_plot_not_log.png")