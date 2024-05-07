# src/modules/autoencodermodules/pose_encoder.py
import torch.nn as nn
from spatial_vae.models import SpatialGenerator

POSE_DIM = 6
LHW_DIM = 3
HIDDEN_DIM_1_DIV = 8
HIDDEN_DIM_2_DIV = 4

class PoseEncoder(nn.Module): 
    """
    PoseEncoder is a module that encodes pose features into image features.
    """

    def __init__(self, enc_feat_dims, pose_feat_dims, activation="relu"):
        """
        Initializes a new instance of the PoseEncoder class.

        Args:
            enc_feat_dims (int): Dimensionality of the image features.
            pose_feat_dims (int): Dimensionality of the pose features.
        """
        super(PoseEncoder, self).__init__()
        
        hidden_dim_1 = enc_feat_dims // HIDDEN_DIM_1_DIV
        hidden_dim_2 = enc_feat_dims // HIDDEN_DIM_2_DIV
        
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "softplus":
            activation = nn.Softplus()
        else:
            raise ValueError("Invalid activation function. Please provide a valid activation function in ['relu', 'softplus'].")
        
        self.fc = nn.Sequential(
            nn.Linear(pose_feat_dims, hidden_dim_1),
            activation,
            nn.Linear(hidden_dim_1, hidden_dim_2),
            activation,
            nn.Linear(hidden_dim_2, enc_feat_dims)
        )
        
    def forward(self, x):
        """
        Forward pass of the PoseEncoder module.

        Args:
            x (tensor): Input tensor containing pose features.

        Returns:
            tensor: Output tensor containing encoded image features.
        """
        return self.fc(x)

class PoseEncoderSpatialVAE(SpatialGenerator):
    def __init__(self, num_classes=1, num_channels=16, n=16, m=16, activation="tanh", **kwargs):
        latent_dim = POSE_DIM + LHW_DIM + num_classes # 10 = 6 + 3 + 1
        n_out = num_channels * n * m # 16 * 16 * 16 = 4096
       
        activation = nn.Tanh if activation == "tanh" else nn.ReLU
        
        kwargs.update({  
            "n_out": n_out,
            "latent_dim": latent_dim,
            "activation": activation
        })
        
        super().__init__(**kwargs)
        print("PoseEncoderSpatialVAE kwargs: ", kwargs)
        print("PoseEncoderSpatialVAE expected input shape: ", latent_dim)
        print("PoseEncoderSpatialVAE expected output shape: ", n_out)
        print("PoseEncoderSpatialVAE latent_linear: ", self.latent_linear)
        print("PoseEncoderSpatialVAE layers: ", self.layers)
    
    def forward(self, x):
        print("PoseEncoderSpatialVAE input shape: ", x.shape)
        
        latent_linear_out = self.latent_linear(x)
        out = self.layers(latent_linear_out) 
        print("PoseEncoderSpatialVAE out shape: ", out.shape)
        return out