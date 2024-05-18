# src/modules/autoencodermodules/pose_decoder.py
import torch.nn as nn
import logging
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

POSE_DIM = 4
LHW_DIM = 3
FILL_FACTOR_DIM=1
HIDDEN_DIM_1_DIV = 4
HIDDEN_DIM_2_DIV = 8

class PoseDecoder(nn.Module):
    """
    Decoder module for pose estimation.
    """

    def __init__(self, enc_feat_dims, pose_feat_dims, activation="relu"):
        """
        Initialize the PoseDecoder module.

        Args:
            enc_feat_dims (int): The dimensionality of the input image features.
            pose_feat_dims (int): The dimensionality of the output pose features.
        """
        
        super(PoseDecoder, self).__init__()
        
        hidden_dim_1 = enc_feat_dims // HIDDEN_DIM_1_DIV
        hidden_dim_2 = enc_feat_dims // HIDDEN_DIM_2_DIV
        
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "softplus":
            activation = nn.Softplus()
        else:
            raise ValueError("Invalid activation function. Please provide a valid activation function in ['relu', 'softplus'].")
        
        self.fc = nn.Sequential(
            nn.Linear(enc_feat_dims, hidden_dim_1), #  256, 64
            activation,
            nn.Linear(hidden_dim_1, hidden_dim_2), # 64, 32
            activation,
            nn.Linear(hidden_dim_2, pose_feat_dims) # 32, 10
        )  
        
    def forward(self, x):
        """
        Forward pass of the pose decoder.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        # input shape: torch.Size([8, 4096])
        return self.fc(x)
    
   
class PoseDecoderSpatialVAE(nn.Module):
    def __init__(self, num_classes=2, num_channels=16, n=16, m=16, activation="tanh", **kwargs):
        super(PoseDecoderSpatialVAE, self).__init__()
        n_out = num_channels * n * m # 16 * 16 * 16 = 4096
        inf_dim = ((POSE_DIM + LHW_DIM + FILL_FACTOR_DIM) * 2) + num_classes # (6 + 3 + 1) * 2  + 1 = 21
        activation = nn.Tanh if activation == "tanh" else nn.ReLU
        
        kwargs.update({
            
            "n": n_out,
            "latent_dim": inf_dim,
            "activation": activation,
        })
        latent_dim = inf_dim
        n = n_out
        
        self.latent_dim = latent_dim
        self.n = n
        hidden_dim = kwargs.get("hidden_dim", 500)
        num_layers = kwargs.get("num_layers", 2)
        resid = kwargs.get("resid", False)

        layers = [nn.Linear(n, hidden_dim),
                  activation(),
                 ]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())

        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        # x is (batch,num_coords)
        z = self.layers(x)
        return z