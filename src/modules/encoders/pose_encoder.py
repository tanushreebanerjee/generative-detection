# src/modules/encoders/pose_encoder.py
import torch.nn as nn

class PoseEncoder(nn.Module): 
    """
    PoseEncoder is a module that encodes pose features into image features.
    """

    def __init__(self, config):
        """
        Initializes a new instance of the PoseEncoder class.

        Args:
            config (object): Configuration object containing pose and image feature dimensions.
        """
        super(PoseEncoder, self).__init__()
        self.config = config
        self.fc = nn.Linear(config.pose_feat_dims, config.img_feat_dims)
    
    def forward(self, x):
        """
        Forward pass of the PoseEncoder module.

        Args:
            x (tensor): Input tensor containing pose features.

        Returns:
            tensor: Output tensor containing encoded image features.
        """
        return self.fc(x)
