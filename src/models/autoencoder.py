# src/models/autoencoder.py

from ldm.models.autoencoder import AutoencoderKL
from src.modules.autoencodermodules.feat_encoder import FeatEncoder
from src.modules.autoencodermodules.feat_decoder import FeatDecoder
from src.modules.autoencodermodules.pose_encoder import PoseEncoder
from src.modules.autoencodermodules.pose_decoder import PoseDecoder
from ldm.util import instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import random
from math import radians
import numpy as np
import logging
from src.util.pose_transforms import euler_angles_translation2se3_log_map

POSE_6D_DIM = 6
PITCH_MAX = 360
YAW_MAX = 30

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
                 euler_convention,
                 ckpt_path=None,
                 ignore_keys=[],
                 image1rgb_key="image1_rgb",
                 pose1_key="pose1",
                 image1mask_key="image1_mask",
                 image2rgb_key="image2_rgb",
                 pose2_key="pose2",
                 image2mask_key="image2_mask",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        pl.LightningModule.__init__(self)
        self.image1rgb_key = image1rgb_key
        self.pose1_key = pose1_key
        self.image1mask_key = image1mask_key
        self.image2rgb_key = image2rgb_key
        self.pose2_key = pose2_key
        self.image2mask_key = image2mask_key
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
                       "pose_feat_dims": POSE_6D_DIM}
                
        self.pose_decoder = PoseDecoder(**feat_dim_config)
        self.pose_encoder = PoseEncoder(**feat_dim_config)
        self.z_channels = ddconfig["z_channels"]
        self.euler_convention=euler_convention
    
    def _get_img_feat_dims(self, ddconfig):
        """ pass in dummy input of size from config to get the output size of encoder and quant_conv """
        batch_size = 1
        dummy_input = torch.randn(batch_size, ddconfig["in_channels"], ddconfig["resolution"], ddconfig["resolution"])
        h = self.encoder(dummy_input)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        img_feat_map = posterior.sample()
        img_feat_map_flat = img_feat_map.view(img_feat_map.size(0), -1)
        return img_feat_map_flat.size(1)
        
    def _decode_pose(self, x):
        """
        Decode the pose from the given image feature map.
        
        Args:
            x: Input image feature map tensor.
        
        Returns:
            Decoded pose tensor.
        """
        x = x.view(x.size(0), -1)  # flatten the input tensor
        return self.pose_decoder(x)
    
    def _encode_pose(self, x):
        """
        Encode the pose to get the pose feature map from the given pose tensor.
        
        Args:
            x: Input pose tensor.
        
        Returns:
            Encoded pose feature map tensor.
        """
        flattened_encoded_pose_feat_map = self.pose_encoder(x)        
        return flattened_encoded_pose_feat_map.view(flattened_encoded_pose_feat_map.size(0), self.z_channels, 
                                                    int(math.sqrt(flattened_encoded_pose_feat_map.shape[1]//self.z_channels)), 
                                                    int(math.sqrt(flattened_encoded_pose_feat_map.shape[1]//self.z_channels)))
    
    def _get_feat_map_img_pose(self, img_feat_map):
            """
            Get the feature map of image and pose.

            Args:
                img_feat_map (Tensor): The feature map of the image.

            Returns:
                feat_map_img_pose (Tensor): The combined feature map of image and pose.
                pose_decoded (Tensor): The decoded pose.

            """
            pose_decoded = self._decode_pose(img_feat_map)
            pose_feat_map = self._encode_pose(pose_decoded)         
            feat_map_img_pose = img_feat_map + pose_feat_map
            pose_decoded_reshaped = pose_decoded.reshape(pose_decoded.size(0), -1)
            return feat_map_img_pose, pose_decoded_reshaped
    
    def _get_img2_reconstruction(self, pose2, z1):
            """
            Reconstructs the second image based on the given pose and latent code.

            Args:
                pose2 (Tensor): The pose of the second image.
                z1 (Tensor): The latent code of the first image.

            Returns:
                Tensor: The reconstructed second image.
            """
            pose2 = pose2.reshape(pose2.size(0), -1)
            pose_feat_map = self._encode_pose(pose2)  
            feat_map_img_pose = z1 + pose_feat_map
            dec2 = self.decode(feat_map_img_pose)
            return dec2
    
    def forward(self, input1, pose2, sample_posterior=True):
        """
        Forward pass of the autoencoder.
        
        Args:
            input: Input image tensor.
            sample_posterior: Whether to sample from the posterior distribution.
        
        Returns:
            tuple: Tuple containing 
                - dec: Decoded image tensor.
                - posterior: Posterior distribution.
                - pose_decoded: Decoded pose tensor.
        """
        posterior1 = self.encode(input1)
        if sample_posterior:
            img_feat_map1 = posterior1.sample()
        else:
            img_feat_map1 = posterior1.mode()
        
        dec2 = self._get_img2_reconstruction(pose2, img_feat_map1)
        
        feat_map_img_pose1, pose_decoded1 = self._get_feat_map_img_pose(img_feat_map1)
            
        dec1 = self.decode(feat_map_img_pose1)
        
        return dec1, dec2, posterior1, pose_decoded1
        
    def get_pose_input(self, batch, k):
        x = batch[k] 
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs1_rgb = self.get_input(batch, self.image1rgb_key)
        pose_inputs1 = self.get_pose_input(batch, self.pose1_key)
        inputs1_mask = self.get_input(batch, self.image1mask_key)
        
        inputs2_rgb = self.get_input(batch, self.image2rgb_key)
        pose_inputs2 = self.get_pose_input(batch, self.pose2_key)
        inputs2_mask = self.get_input(batch, self.image2mask_key)
        
        reconstructions1, reconstructions2, posterior1, pose_reconstructions1 = self(inputs1_rgb, pose_inputs2)
        
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs1_rgb, inputs2_rgb,
                                            inputs1_mask, inputs2_mask, 
                                            reconstructions1, reconstructions2, 
                                            pose_inputs1, pose_reconstructions1,
                                            posterior1, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs1_rgb, inputs2_rgb, 
                                                inputs1_mask, inputs2_mask, 
                                                reconstructions1, reconstructions2, 
                                                pose_inputs1, pose_reconstructions1,
                                                posterior1, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        inputs1_rgb = self.get_input(batch, self.image1rgb_key) # torch.Size([8, 3, 64, 64])
        pose_inputs_1 = self.get_pose_input(batch, self.pose1_key) # torch.Size([8, 1, 6])
        inputs1_mask = self.get_input(batch, self.image1mask_key)
        inputs2_rgb = self.get_input(batch, self.image2rgb_key)
        pose_inputs_2 = self.get_pose_input(batch, self.pose2_key)
        inputs2_mask = self.get_input(batch, self.image2mask_key)
        reconstructions1, reconstructions2, posterior1, pose_reconstructions1 = self(inputs1_rgb, pose_inputs_2)
        
        _, log_dict_ae = self.loss(inputs1_rgb, inputs2_rgb, 
                                   inputs1_mask, inputs2_mask, 
                                   reconstructions1, reconstructions2, 
                                   pose_inputs_1, pose_reconstructions1,
                                   posterior1, 0, self.global_step,
                                   last_layer=self.get_last_layer(), split="val")

        _, log_dict_disc = self.loss(inputs1_rgb, inputs2_rgb, 
                                     inputs1_mask, inputs2_mask, 
                                     reconstructions1, reconstructions2, 
                                     pose_inputs_1, pose_reconstructions1,
                                     posterior1, 1, self.global_step,
                                     last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.pose_encoder.parameters())+
                                  list(self.pose_decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    
    def _perturb_poses(self, pose_inputs, pitch_max=PITCH_MAX, yaw_max=YAW_MAX):
        batch_size = pose_inputs.size(0)        
        rotation_deg = torch.tensor([random.uniform(0, pitch_max) for _ in range(batch_size)])
        elevation_deg = torch.tensor([random.uniform(0, yaw_max) for _ in range(batch_size)])
        rotation_rad = torch.tensor([radians(i) for i in rotation_deg]) 
        elevation_rad = torch.tensor([radians(i) for i in elevation_deg])
        
        roll = torch.zeros(batch_size)
        pitch = elevation_rad
        yaw = rotation_rad
        
        if self.euler_convention == "ZYX":
            euler_angles = torch.stack([yaw, pitch, roll])
        elif self.euler_convention == "XYZ":
            euler_angles = torch.stack([roll, pitch, yaw])
        else:
            raise ValueError(f"Invalid convention: {self.euler_convention}, must be either ZYX or XYZ")   
        euler_angle = torch.stack([yaw, pitch, roll])
        translation = torch.zeros(batch_size, 3)
        pose_6d = euler_angles_translation2se3_log_map(euler_angle, translation, self.euler_convention)
        return pose_6d.to(self.device)

    def _get_feat_map_img_perturbed_pose(self, img_feat_map, pose_decoded_perturbed):
        pose_feat_map = self._encode_pose(pose_decoded_perturbed)         
        feat_map_img_pose = img_feat_map + pose_feat_map
        return feat_map_img_pose
    
    def _perturbed_pose_forward(self, posterior, pose_decoded, sample_posterior=True):
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        pose_decoded_perturbed = self._perturb_poses(pose_decoded).reshape(pose_decoded.size(0), -1)
        z = self._get_feat_map_img_perturbed_pose(z, pose_decoded_perturbed)
        dec = self.decode(z)
        return dec
            
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        
        # x = self.get_input(batch, self.image_key)
        x1_rgb = self.get_input(batch, self.image1rgb_key)
        x2_rgb = self.get_input(batch, self.image2rgb_key)
        x1_mask = self.get_input(batch, self.image1mask_key)
        x2_mask = self.get_input(batch, self.image2mask_key)
        pose2 = self.get_pose_input(batch, self.pose2_key)
        
        # x = x.to(self.device)
        x1_rgb = x1_rgb.to(self.device)
        x2_rgb = x2_rgb.to(self.device)
        pose2 = pose2.to(self.device)
        x1_mask = x1_mask.to(self.device)
        x2_mask = x2_mask.to(self.device)
        # convert mask to float, 0.0, 1.0 if not already
        if x1_mask.dtype != torch.float32:
            x1_mask = x1_mask.float()
        if x2_mask.dtype != torch.float32:
            x2_mask = x2_mask.float()
            
        if not only_inputs:
            xrec1, xrec2, posterior1, pose_decoded1 = self(x1_rgb, pose2)
            xrec1_perturbed_pose = self._perturbed_pose_forward(posterior1, pose_decoded1)
            
            xrec1_rgb = xrec1[:, :3, :, :]
            xrec2_rgb = xrec2[:, :3, :, :]
            xrec1_perturbed_pose_rgb = xrec1_perturbed_pose[:, :3, :, :]
            logging.info(f"xrec1_rgb shape: {xrec1_rgb.shape}, xrec2_rgb shape: {xrec2_rgb.shape}, xrec1_perturbed_pose_rgb shape: {xrec1_perturbed_pose_rgb.shape}")
            if xrec1.shape[1] == 4: # alpha channel prediction loggging
                
                xrec1_mask = np.array(xrec1[:, 3, :, :].cpu())
                xrec2_mask = np.array(xrec2[:, 3, :, :].cpu())
                xrec1_perturbed_pose_mask = np.array(xrec1_perturbed_pose[:, 3, :, :].cpu())
                logging.info(f"xrec1_mask shape: {xrec1_mask.shape}, xrec2_mask shape: {xrec2_mask.shape}, xrec1_perturbed_pose_mask shape: {xrec1_perturbed_pose_mask.shape}")

                # convert to uint8, in height, width format (gray scale) for PIL
                xrec1_mask = np.clip(xrec1_mask.cpu().numpy(), 0., 1.)
                xrec2_mask = np.clip(xrec2_mask.cpu().numpy(), 0., 1.)
                xrec1_perturbed_pose_mask = np.clip(xrec1_perturbed_pose_mask.cpu().numpy(), 0., 1.)
                
                xrec1_mask = (xrec1_mask * 255).astype(np.uint8).transpose(1, 2, 0)
                xrec2_mask = (xrec2_mask * 255).astype(np.uint8).transpose(1, 2, 0)
                xrec1_perturbed_pose_mask = (xrec1_perturbed_pose_mask * 255).astype(np.uint8).transpose(1, 2, 0)
                logging.info(f"xrec1_mask shape: {xrec1_mask.shape}, xrec2_mask shape: {xrec2_mask.shape}, xrec1_perturbed_pose_mask shape: {xrec1_perturbed_pose_mask.shape}")
                log["reconstructions1_mask"] = xrec1_mask
                log["reconstructions2_mask"] = xrec2_mask
                log["perturbed_pose_reconstruction_mask"] = xrec1_perturbed_pose_mask
            
            if x1_rgb.shape[1] > 3:
                # colorize with random projection
                assert xrec1_rgb.shape[1] > 3
                x1_rgb = self.to_rgb(x1_rgb)
                x2_rgb = self.to_rgb(x2_rgb)
                
                xrec1_rgb = self.to_rgb(xrec1_rgb)
                xrec2_rgb = self.to_rgb(xrec2_rgb)
                xrec1_perturbed_pose_rgb = self.to_rgb(xrec1_perturbed_pose_rgb)

            log["samples1"] = self.decode(torch.randn_like(posterior1.sample()))
            log["reconstructions1_rgb"] = xrec1_rgb
            log["reconstructions2_rgb"] = xrec2_rgb
            log["perturbed_pose_reconstruction_rgb"] = xrec1_perturbed_pose_rgb
        
        log["inputs1_rgb"] = x1_rgb
        log["inputs2_rgb"] = x2_rgb
        log["inputs1_mask"] = x1_mask
        log["inputs2_mask"] = x2_mask
        return log
    
    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
