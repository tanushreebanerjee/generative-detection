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
                 image_rgb_key="image1_rgb",
                 pose_key="pose",
                 image_mask_key="image_mask",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        pl.LightningModule.__init__(self)
        self.image_rgb_key = image_rgb_key
        self.pose_key = pose_key
        self.image_mask_key = image_mask_key
        self.encoder = FeatEncoder(**ddconfig)
        self.decoder = FeatDecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv_obj = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        sefl.quant_conv_pose = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1) # TODO: Need to fix the dimensions
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
    
    def encode(self, x):
        h = self.encoder(x)
        moments_obj = self.quant_conv_obj(h)
        moments_pose = self.quant_conv_pose(h)
        posterior_obj = DiagonalGaussianDistribution(moments_obj)
        posterior_pose = DiagonalGaussianDistribution(moments_pose)
        return posterior_obj, posterior_pose
    
    def forward(self, input, sample_posterior=True):
            """
            Forward pass of the autoencoder model.
            
            Args:
                input (Tensor): Input tensor to the autoencoder.
                sample_posterior (bool, optional): Whether to sample from the posterior distribution or use the mode.
            
            Returns:
                dec_obj (Tensor): Decoded object tensor.
                dec_pose (Tensor): Decoded pose tensor.
                posterior_obj (Distribution): Posterior distribution of the object latent space.
                posterior_pose (Distribution): Posterior distribution of the pose latent space.
            """
            posterior_obj, posterior_pose = self.encode(input1)
            if sample_posterior:
                z_obj = posterior_obj.sample()
                z_pose = posterior_pose.sample()
            else:
                z_obj = posterior_obj.mode()
                z_pose = posterior_pose.mode()
            
            dec_pose = self._decode_pose(z_pose)
            enc_pose = self._encode_pose(dec_pose)
            
            z_obj_pose = z_obj + enc_pose
            
            dec_obj = self.decode(z_obj_pose)
            
            return dec_obj, dec_pose, posterior_obj, posterior_pose
        
    def get_pose_input(self, batch, k):
        x = batch[k] 
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        rgb_gt = self.get_input(batch, self.image_rgb_key)
        pose_gt = self.get_pose_input(batch, self.pose_key)
        mask_gt = self.get_input(batch, self.image_mask_key)
         
        dec_obj, dec_pose, posterior_obj, posterior_pose = self(rgb_gt)
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(rgb_gt, mask_gt, pose_gt
                                            dec_obj, dec_pose
                                            posterior_obj, posterior_pose, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(rgb_gt, mask_gt, pose_gt
                                                dec_obj, dec_pose
                                                posterior_obj, posterior_pose, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        
        rgb_gt = self.get_input(batch, self.image_rgb_key)
        pose_gt = self.get_pose_input(batch, self.pose_key)
        mask_gt = self.get_input(batch, self.image_mask_key)
         
        dec_obj, dec_pose, posterior_obj, posterior_pose = self(rgb_gt)
       
        
        _, log_dict_ae = self.loss(rgb_gt, mask_gt, pose_gt
                                      dec_obj, dec_pose
                                      posterior_obj, posterior_pose, 0, self.global_step,
                                      last_layer=self.get_last_layer(), split="val")

        _, log_dict_disc = self.loss(rgb_gt, mask_gt, pose_gt
                                        dec_obj, dec_pose
                                        posterior_obj, posterior_pose, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv_obj.parameters())+
                                  list(self.quant_conv_pose.parameters())+
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
 
    def _perturbed_pose_forward(self, posterior_obj, dec_pose, sample_posterior=True):
        if sample_posterior:
            z_obj = posterior_obj.sample()
        else:
            z_obj = posterior_obj.mode()
        dec_pose_perturbed = self._perturb_poses(dec_pose).reshape(dec_pose.size(0), -1)
        enc_pose_perturbed = self._encode_pose(dec_pose_perturbed)
        z_obj_pose_perturbed = z_obj + enc_pose_perturbed
        dec_obj_pose_perturbed = self.decode(z_obj_pose_perturbed)
        return dec_obj_pose_perturbed
            
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        
        # x = self.get_input(batch, self.image_key)
        x_rgb = self.get_input(batch, self.image_rgb_key)
        x_mask = self.get_input(batch, self.image_mask_key)
        
        # x = x.to(self.device)
        x_rgb = x_rgb.to(self.device)
        x_mask = x_mask.to(self.device)
        
        # convert mask to float, 0.0, 1.0 if not already
        if x_mask.dtype != torch.float32:
            x_mask = x_mask.float()
            
        if not only_inputs:
            
            xrec, poserec, posterior_obj, posterior_pose = self(x_rgb)
            xrec_perturbed_pose = self._perturbed_pose_forward(posterior_obj, poserec)
            
            xrec_rgb = xrec1[:, :3, :, :]
            xrec_perturbed_pose_rgb = xrec1_perturbed_pose[:, :3, :, :]
            if x_rgb.shape[1] > 3:
                # colorize with random projection
                assert xrec_rgb.shape[1] > 3
                x_rgb = self.to_rgb(x_rgb)
                xrec_rgb = self.to_rgb(xrec_rgb)
                xrec_perturbed_pose_rgb = self.to_rgb(xrec_perturbed_pose_rgb)

            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions_rgb"] = torch.tensor(xrec_rgb)
            log["perturbed_pose_reconstruction_rgb"] = torch.tensor(xrec_perturbed_pose_rgb)
        
        log["inputs_rgb"] = x_rgb
        log["inputs_mask"] = x_mask
        return log
    
    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
