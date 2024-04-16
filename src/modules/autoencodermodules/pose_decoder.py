# src/modules/autoencodermodules/pose_decoder.py
import torch.nn as nn
import logging

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
        
        hidden_dim_1 = enc_feat_dims // 4
        hidden_dim_2 = enc_feat_dims // 8
        
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "softplus":
            activation = nn.Softplus()
        else:
            raise ValueError("Invalid activation function. Please provide a valid activation function in ['relu', 'softplus'].")
        
        self.fc = nn.Sequential(
            nn.Linear(enc_feat_dims, hidden_dim_1),
            activation,
            nn.Linear(hidden_dim_1, hidden_dim_2),
            activation,
            nn.Linear(hidden_dim_2, pose_feat_dims)
        )
        
        logging.info("PoseDecoder initialized.")
        logging.info(self.fc)
        
        
    def forward(self, x):
        """
        Forward pass of the pose decoder.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(x)