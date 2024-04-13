# src/modules/autoencodermodules/pose_decoder.py
import torch.nn as nn

class PoseDecoder(nn.Module):
    """
    Decoder module for pose estimation.
    """

    def __init__(self, enc_feat_dims, pose_feat_dims):
        """
        Initialize the PoseDecoder module.

        Args:
            enc_feat_dims (int): The dimensionality of the input image features.
            pose_feat_dims (int): The dimensionality of the output pose features.
        """
        
        super(PoseDecoder, self).__init__()
        self.fc = nn.Linear(enc_feat_dims, pose_feat_dims)
    
    def forward(self, x):
        """
        Forward pass of the pose decoder.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(x)