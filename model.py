import torch
import lightning
from torch import nn
from dataclasses import dataclass   

#TODO: Add support for mare activation functions
@dataclass
class ModelConfig:
    input_dim: int
    hidden_dims: list[int]
    backbone_output_dim: int
    small_num_basis_funcs: int
    big_num_basis_funcs: int
    output_dim: int
    activation: str = "relu"
    small_model_save_path: str = "small_model.pth"
    big_model_save_path: str = "big_model.pth"

    def __post_init__(self):
        if self.activation == "relu":
            self.activation_fn = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")


class FeatureBackbone(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_fn: nn.Module,
    ):
        super(FeatureBackbone, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class SmallModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SmallModel, self).__init__()
        self.backbone = FeatureBackbone(
            config.input_dim,
            config.hidden_dims,
            config.backbone_output_dim,
            config.activation_fn,
        )
        self.feature_projector = nn.Linear(config.backbone_output_dim, config.small_num_basis_funcs)
        self.theta = nn.Linear(config.small_num_basis_funcs, 1)
        self.activation_fn = config.activation_fn()

    def get_features(self, x):
        features = self.backbone(x)
        projected_features = self.activation_fn(self.feature_projector(features))
        return projected_features

    def forward(self, x):
        projected_features = self.get_features(x)
        output = self.theta(projected_features)
        return output   

# TODO: Big model should be able to optionally accept a pre-trained small model not just load from weights
class BigModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(BigModel, self).__init__()
        self.config = config
        small_model = SmallModel(config)
        state = torch.load(config.small_model_save_path, map_location="cpu")
        small_model.load_state_dict(state)

        for p in small_model.parameters():
            p.requires_grad = False

        self.backbone = small_model.backbone
        self.feature_backbone = FeatureBackbone(
            config.backbone_output_dim,
            config.hidden_dims,
            config.backbone_output_dim,
            config.activation_fn,
        )
        self.mtm_projector = small_model.feature_projector
    
        self.mtu_projector = nn.Linear(
            config.backbone_output_dim, 
            config.big_num_basis_funcs - config.small_num_basis_funcs
        )

        self.theta = nn.Linear(config.big_num_basis_funcs, 1)
        self.activation_fn = config.activation_fn()

    def forward(self, x):
        all_projected = self.get_features(x)
        output = self.theta(all_projected)
        return output

    def get_features(self, x):
        features = self.backbone(x)
        small_projected = self.activation_fn(self.mtm_projector(features))
        richer_features = self.feature_backbone(features)
        big_projected = self.activation_fn(self.mtu_projector(richer_features))
        all_projected = torch.hstack([small_projected, big_projected])
        return all_projected
    
    def compute_gad(self, x, num_train) -> dict:
        # X should be ALL of the data
        projected = self.get_features(x)

        # Remember to add the bias term
        projected = torch.hstack((
            torch.ones(projected.shape[0], 1), projected
        ))

        m_tm = projected[:num_train, : self.config.small_num_basis_funcs + 1]
        m_tu = projected[:num_train, self.config.small_num_basis_funcs + 1 :]

        m_tm_pinv = torch.linalg.pinv(m_tm)
        a = m_tm_pinv @ m_tu
        b = m_tm_pinv @ m_tm
        p_n = torch.eye(b.shape[1], device=x.device) - b

        theta_m = torch.vstack((
            self.theta.bias.unsqueeze(0), self.theta.weight[:, : self.config.small_num_basis_funcs].T
        ))

        theta_u = self.theta.weight[:, self.config.small_num_basis_funcs :].T

        return {
            "m_tm": m_tm,
            "m_tu": m_tu,
            "a": a,
            "p_n": p_n,
            "theta_m": theta_m,
            "theta_u": theta_u,
        }
        
    
    def _get_unmodeled_param_norm(self):
        return torch.norm(
            self.theta.weight[:, self.config.small_num_basis_funcs:], 
            p=2,
        )
    
class Model(lightning.LightningModule):
    def __init__(
        self, 
        config: ModelConfig, 
        model_type: str
    ):
        super(Model, self).__init__()
        if model_type == "small":
            self.model = SmallModel(config)
        elif model_type == "big":
            self.model = BigModel(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.criterion = nn.MSELoss()
        self.final_val_loss = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.final_val_loss = loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def get_features(self, x):
        return self.model.get_features(x)
    
    def _get_unmodeled_param_norm(self):
        if isinstance(self.model, BigModel):
            return self.model._get_unmodeled_param_norm()
        else:
            raise ValueError("Unmodeled parameter norm is only defined for BigModel")
        
    
    


