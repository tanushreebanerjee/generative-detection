# src/modules/autoencodermodules/pose_encoder.py
import torch.nn as nn
from spatial_vae.models import SpatialGenerator
from src.util.misc import swish_activation as swish
import numpy as np
import torch.nn.functional as F

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
    def __init__(self, num_classes=1, num_channels=16, n=16, m=16, activation="swish", **kwargs):
        latent_dim = POSE_DIM + LHW_DIM + num_classes # 10 = 6 + 3 + 1
        n_out = num_channels * n * m # 16 * 16 * 16 = 4096
        if activation == "swish":
            activation = swish
        elif activation == "tanh":
            activation = nn.Tanh
        else:
            activation = nn.ReLU
        
        kwargs.update({  
            "n_out": n_out,
            "latent_dim": latent_dim,
            "activation": activation
        })
        
        super().__init__(**kwargs)
        
        ## x coordinate array
        xgrid = np.linspace(-1, 1, m)
        ygrid = np.linspace(1, -1, n)
        x0,x1 = np.meshgrid(xgrid, ygrid)
        x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
        x_coord = torch.from_numpy(x_coord).float()
        x = x_coord
        x = x.expand(b, x.size(0), x.size(1))
        self.x = x.contiguous()
    
    def forward(self, z):
        # x is (batch, num_coords, 2)
        # z is (batch, latent_dim)
        x = self.x
        
        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        b = x.size(0)
        n = x.size(1)
        x = x.view(b*n, -1)
        # if self.expand_coords:
        #     x2 = x**2
        #     xx = x[:,0]*x[:,1]
        #     x = torch.cat([x, x2, xx.unsqueeze(1)], 1)

        h_x = self.coord_linear(x)
        h_x = h_x.view(b, n, -1)

        h_z = 0
        if hasattr(self, 'latent_linear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            h_z = h_z.unsqueeze(1)

        # h_bi = 0
        # if hasattr(self, 'bilinear'):
        #     if len(z.size()) < 2:
        #         z = z.unsqueeze(0)
        #     z = z.unsqueeze(1) # broadcast over coordinates
        #     x = x.view(b, n, -1)
        #     z = z.expand(b, x.size(1), z.size(2)).contiguous()
        #     h_bi = self.bilinear(x, z)

        h = h_x + h_z #+ h_bi # (batch, num_coords, hidden_dim)
        h = h.view(b*n, -1)

        y = self.layers(h) # (batch*num_coords, nout)
        y = y.view(b, n, -1)

        # if self.softplus: # only apply softplus to first output
        #     y = torch.cat([F.softplus(y[:,:,:1]), y[:,:,1:]], 2)
        return y