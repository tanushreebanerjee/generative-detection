# src/modules/autoencodermodules/pose_encoder.py
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Variable

POSE_DIM = 4
LHW_DIM = 3
FILL_FACTOR_DIM=1
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

class PoseEncoderSpatialVAE(nn.Module):
    def __init__(self, num_classes=2, num_channels=16, n=16, m=16, activation="swish", hidden_dim=500, num_layers=2):
        super(PoseEncoderSpatialVAE, self).__init__()
        latent_dim = POSE_DIM + LHW_DIM + FILL_FACTOR_DIM + num_classes # 10 = 6 + 3 + 1
        n_out = num_channels * n * m # 16 * 16 * 16 = 4096
        if activation == "swish":
            activation = nn.SiLU
        elif activation == "tanh":
            activation = nn.Tanh
        else:
            activation = nn.ReLU
            
        self.num_channels = num_channels # 16
        self.n = n # 16
        self.m = m # 16
        self.in_dim = 2 # 2 dim position: x, y
        self.num_coords = n * m # 16 * 16 = 256
        self.feat_size = 4 # TODO: try 10
        self.h_dim = self.num_coords * self.feat_size # 16 * 16 * 4 = 1024
        self.x_dim = self.in_dim * self.num_coords # 16, 16, 2 = 512
        self.z_dim = latent_dim # 10
        
        # x --> h_x
        self.coord_linear = nn.Linear(self.x_dim, self.h_dim) # (512, 1024)
        
        if latent_dim > 0:
            self.latent_linear = nn.Linear(self.z_dim, self.feat_size, bias=False) # (10, 4)

        layers = [activation()]
        for layer_id in range(1,num_layers):
            
            if layer_id == 1: 
                layer = nn.Linear(self.h_dim, hidden_dim) # (1024, 500)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim) # (500, 500)
            layers.append(layer)
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, n_out)) # (500, 4096)

        self.layers = nn.Sequential(*layers) # h --> y
        
        ## x coordinate array
        xgrid = np.linspace(-1, 1, m) 
        ygrid = np.linspace(1, -1, n)
        x0,x1 = np.meshgrid(xgrid, ygrid)
        x_coord = np.stack([x0.ravel(), x1.ravel()], 1) # (256, 2)
        x_coord = torch.from_numpy(x_coord).float() 
        x_coord = x_coord
        
        self.x = Variable(x_coord)
        
    
    def forward(self, z):
        b = z.size(0) # 4
        x = self.x.expand(b, self.num_coords, self.in_dim).to(z) # (batch, num_coords, 2) 
        x = x.contiguous()
        if len(x.size()) < 3:
            x = x.unsqueeze(0) # (1, 16*16, 2)
        
        x = x.view(b, self.num_coords * self.in_dim) # (batch, num_coords*2) = (4, 512)
        h_x = self.coord_linear(x) # h_x torch.Size([4, 1024]) = b, 16 * 16 * 4
        
        if len(z.size()) < 2:
            z = z.unsqueeze(0) # (1, 10)
        h_z = self.latent_linear(z) # h_z torch.Size([4, 4]) = b, 4
        
        # target size: # (4, 16, 16, 4). use tensor expand
        h_z = h_z.unsqueeze(1).expand(b, self.num_coords, self.feat_size) # (4, 256, 4)
        h_z = h_z.reshape(b, self.num_coords * self.feat_size) # (4, 1024)
        
        h = h_x + h_z # h torch.Size([4, 1024])
        y = self.layers(h) # y torch.Size([4, 4096])
        return y
    