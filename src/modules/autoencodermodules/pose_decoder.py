# src/modules/autoencodermodules/pose_decoder.py
import torch.nn as nn
import logging
from spatial_vae.models import InferenceNetwork
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

POSE_DIM = 6
LHW_DIM = 3
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
    
   
class PoseDecoderSpatialVAE(InferenceNetwork):
    def __init__(self, num_classes=1, num_channels=16, n=16, m=16, activation="tanh", **kwargs):
        
        n_out = num_channels * n * m
        inf_dim = ((POSE_DIM + LHW_DIM) * 2) + num_classes
        activation = nn.Tanh if activation == "tanh" else nn.ReLU
        
        kwargs.update({
            
            "n": n_out,
            "latent_dim": inf_dim,
            "activation": activation,
        })
        
        super().__init__(**kwargs)
        
        print("Expected input shape: (batch_size, num_channels, n, m)", (None, num_channels, n, m))
        print("Expected output shape: (batch_size, inf_dim)", (None, inf_dim))
        
    def forward(self, x):
        # x is (batch,num_coords)
        z = self.layers(x)
        return z