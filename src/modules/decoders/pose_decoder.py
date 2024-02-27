# src/modules/decoders/pose_decoder.py
import torch.nn as nn

class PoseDecoder(nn.Module):
    """
    Decoder module for pose estimation.
    """

    def __init__(self, config):
        super(PoseDecoder, self).__init__()
        self.config = config
        self.fc = nn.Linear(config.img_feat_dims, config.pose_feat_dims)
    
    def forward(self, x):
        """
        Forward pass of the pose decoder.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(x)