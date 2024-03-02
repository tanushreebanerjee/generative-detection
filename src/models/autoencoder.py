# src/models/autoencoder.py

from ldm.models.autoencoder import AutoencoderKL
from src.modules.autoencodermodules.feat_encoder import FeatEncoder
from src.modules.autoencodermodules.feat_decoder import FeatDecoder
from src.modules.autoencodermodules.pose_encoder import PoseEncoder
from src.modules.autoencodermodules.pose_decoder import PoseDecoder
from ldm.util import instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import torch
import logging
import math

SE3_DIM = 16

class Autoencoder(AutoencoderKL):
    """Autoencoder model with KL divergence loss."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class PoseAutoencoder(AutoencoderKL):
    """
    Autoencoder model for pose encoding and decoding.
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__(ddconfig, lossconfig, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor)
        self.image_key = image_key
        self.encoder = FeatEncoder(**ddconfig)
        self.decoder = FeatDecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        img_feat_dims = self._get_img_feat_dims(ddconfig)
        
        feat_dim_config = {"img_feat_dims": img_feat_dims,
                       "pose_feat_dims": SE3_DIM}
                
        self.pose_decoder = PoseDecoder(**feat_dim_config)
        self.pose_encoder = PoseEncoder(**feat_dim_config)
        self.z_channels = ddconfig["z_channels"]
    
    def _get_img_feat_dims(self, ddconfig):
        """ pass in dummy input of size from config to get the output size of encoder and quant_conv """
        batch_size = 1
        dummy_input = torch.randn(batch_size, ddconfig["in_channels"], ddconfig["resolution"], ddconfig["resolution"])
        h = self.encoder(dummy_input)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        img_feat_map = posterior.sample()
        img_feat_map_flat = img_feat_map.view(img_feat_map.size(0), -1)
        logging.info(f"img_feat_map.size(): {img_feat_map.size()}") # torch.Size([1, 64, 2, 2])
        logging.info(f"img_feat_map_flat.size(): {img_feat_map_flat.size()}") #  torch.Size([1, 256])
        return img_feat_map_flat.size(1)
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
        
    def decode_pose(self, x):
        """
        Decode the pose from the given image feature map.
        
        Args:
            x: Input image feature map tensor.
        
        Returns:
            Decoded pose tensor.
        """
        x = x.view(x.size(0), -1)  # flatten the input tensor
        return self.pose_decoder(x)
    
    def encode_pose(self, x):
        """
        Encode the pose to get the pose feature map from the given pose tensor.
        
        Args:
            x: Input pose tensor.
        
        Returns:
            Encoded pose feature map tensor.
        """
        # batch size = 12
        flattened_encoded_pose_feat_map = self.pose_encoder(x) # torch.Size([12, 256])
        # unflatten the tensor to [12, 64, 2, 2])
        return flattened_encoded_pose_feat_map.view(flattened_encoded_pose_feat_map.size(0), self.z_channels, int(math.sqrt(self.embed_dim)), int(math.sqrt(self.embed_dim)))
    
    def forward(self, input, sample_posterior=True):
        """
        Forward pass of the autoencoder.
        
        Args:
            input: Input image tensor.
            sample_posterior: Whether to sample from the posterior distribution.
        
        Returns:
            tuple: Tuple containing 
                - decoded image feature map
                - posterior distribution
                - feature map of image and pose
                - decoded pose
        """
        posterior = self.encode(input)
        if sample_posterior:
            img_feat_map = posterior.sample()
        else:
            img_feat_map = posterior.mode()
            
        ## LDM repo legacy
        z = img_feat_map
        logging.info(f"z.size(): {z.size()}")
        dec = self.decode(z)
        logging.info(f"dec.size(): {dec.size()}")
        ##
        
        logging.info(f"img_feat_map.size(): {img_feat_map.size()}") #  torch.Size([12, 64, 2, 2])
        pose_decoded = self.decode_pose(img_feat_map)
        logging.info(f"pose_decoded.size(): {pose_decoded.size()}") # torch.Size([12, 16])
        pose_encoded = self.encode_pose(pose_decoded)
        logging.info(f"pose_encoded.size(): {pose_encoded.size()}") # torch.Size([12, 256])
         
        pose_feat_map = self.decode(pose_encoded)
        logging.info(f"pose_feat_map.size(): {pose_feat_map.size()}")
        feat_map_img_pose = img_feat_map + pose_feat_map
        logging.info(f"feat_map_img_pose.size(): {feat_map_img_pose.size()}")
        
        
        
        return dec, posterior#, feat_map_img_pose, pose_decoded