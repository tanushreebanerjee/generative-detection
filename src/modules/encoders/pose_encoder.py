# src/modules/encoders/pose_encoder.py
import torch.nn as nn

class PoseEncoder(nn.Module):
    """
    A class representing a pose encoder module.

    Args:
        config (object): Configuration object containing input and output dimensions.

    Attributes:
        config (object): Configuration object containing input and output dimensions.
        fc (nn.Linear): Linear layer for encoding the input.

    Methods:
        forward(x): Forward pass of the pose encoder.

    """
    def __init__(self, config):
        super(PoseEncoder, self).__init__()
        self.config = config
        self.fc = nn.Linear(config.input_dim, config.output_dim)
    
    def forward(self, x):
        return self.fc(x)
