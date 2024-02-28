# src/models/autoencoder.py

from ldm.models.autoencoder import AutoencoderKL
from src.modules.encoders.pose_encoder import PoseEncoder
from src.modules.decoders.pose_decoder import PoseDecoder

class Autoencoder(AutoencoderKL):
    """Autoencoder model with KL divergence loss."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class PoseAutoencoder(AutoencoderKL):
    """
    Autoencoder model for pose encoding and decoding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_encoder = PoseEncoder(*args, **kwargs)
        self.pose_decoder = PoseDecoder(*args, **kwargs)
        
    def decode_pose(self, x):
        """
        Decode the pose from the given input.
        
        Args:
            x: Input tensor.
        
        Returns:
            Decoded pose tensor.
        """
        return self.pose_decoder(x)
    
    def encode_pose(self, x):
        """
        Encode the pose from the given input.
        
        Args:
            x: Input tensor.
        
        Returns:
            Encoded pose tensor.
        """
        return self.pose_encoder(x)
    
    def forward(self, input, sample_posterior=True):
        """
        Forward pass of the autoencoder.
        
        Args:
            input: Input tensor.
            sample_posterior: Whether to sample from the posterior distribution.
        
        Returns:
            Tuple containing the feature map with pose information and the decoded pose.
        """
        posterior = self.encode(input)
        if sample_posterior:
            feat_map = posterior.sample()
        else:
            feat_map = posterior.mode()
            
        pose_decoded = self.decode_pose(feat_map)
        pose_encoded = self.encode_pose(pose_decoded)
         
        pose_feat_map = self.decode(pose_encoded)
        
        feat_map_pose = feat_map + pose_feat_map
        return feat_map_pose, pose_decoded