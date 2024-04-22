# src/modules/autoencodermodules/pose_decoder.py
import torch.nn as nn
import logging

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
        
        hidden_dim_1 = enc_feat_dims // HIDDEN_DIM_1_DIV # hidden_dim_1: 1024
        hidden_dim_2 = enc_feat_dims // HIDDEN_DIM_2_DIV # hidden_dim_2: 512
        
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "softplus":
            activation = nn.Softplus()
        else:
            raise ValueError("Invalid activation function. Please provide a valid activation function in ['relu', 'softplus'].")
        
        self.fc = nn.Sequential(
            nn.Linear(enc_feat_dims, hidden_dim_1), # (0): Linear(in_features=4096, out_features=1024, bias=True)
            activation,
            nn.Linear(hidden_dim_1, hidden_dim_2), # (2): Linear(in_features=1024, out_features=512, bias=True)
            activation,
            nn.Linear(hidden_dim_2, pose_feat_dims) # (4): Linear(in_features=512, out_features=10, bias=True)
        )  
        
    def forward(self, x):
        """
        Forward pass of the pose decoder.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(x)