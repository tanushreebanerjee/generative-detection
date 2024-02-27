# src/modules/decoders/pose_decoder.py
import torch.nn as nn

class PoseDecoder(nn.Module):
    """
    A class representing a pose decoder module.

    Args:
        config (object): Configuration object containing input and output dimensions.

    Attributes:
        config (object): Configuration object containing input and output dimensions.
        fc (nn.Linear): Linear layer for decoding the input.

    Methods:
        forward(x): Forward pass of the pose decoder.

    """
    def __init__(self, config):
        super(PoseDecoder, self).__init__()
        self.config = config
        self.fc = nn.Linear(config.input_dim, config.output_dim)
    
    def forward(self, x):
        return self.fc(x)