# src/models/autoencoder.py

from ldm.models.autoencoder import AutoencoderKL
from src.modules.autoencodermodules.feat_encoder import FeatEncoder
from src.modules.autoencodermodules.feat_decoder import FeatDecoder
from src.modules.autoencodermodules.pose_encoder import PoseEncoderSpatialVAE as PoseEncoder
from src.modules.autoencodermodules.pose_decoder import PoseDecoderSpatialVAE as PoseDecoder
from ldm.util import instantiate_from_config
from src.util.distributions import DiagonalGaussianDistribution
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import random
from math import radians
import numpy as np
import logging

POSE_6D_DIM = 4
PITCH_MAX = 360
YAW_MAX = 60
YAW_MIN = 20
LHW_DIM = 3

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
                 image_mask_key=None,
                 image_rgb_key="patch",
                 pose_key="pose_6d",
                 pose_perturbed_key="pose_6d_perturbed",
                 class_key="class_id",
                 bbox_key="bbox_sizes",
                 colorize_nlabels=None,
                 monitor=None,
                 activation="relu",
                 feat_dims=[16, 16, 16],
                 pose_decoder_config=None,
                 pose_encoder_config=None,
                 ):
        pl.LightningModule.__init__(self)
        self.feature_dims = feat_dims
        self.image_rgb_key = image_rgb_key
        self.pose_key = pose_key
        self.pose_perturbed_key = pose_perturbed_key
        self.class_key = class_key
        self.bbox_key = bbox_key
        self.image_mask_key = image_mask_key
        self.encoder = FeatEncoder(**ddconfig)
        self.decoder = FeatDecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv_obj = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.quant_conv_pose = torch.nn.Conv2d(2*ddconfig["z_channels"], embed_dim, 1) # TODO: Need to fix the dimensions
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.num_classes = lossconfig["params"]["num_classes"]
        enc_feat_dims = self._get_enc_feat_dims(ddconfig)
        
        self.pose_decoder = instantiate_from_config(pose_decoder_config)
        self.pose_encoder = instantiate_from_config(pose_encoder_config)
        self.z_channels = ddconfig["z_channels"]
        self.euler_convention=euler_convention
        self.feat_dims = feat_dims
        
    def _get_enc_feat_dims(self, ddconfig):
        """ pass in dummy input of size from config to get the output size of encoder and quant_conv """
        # multiply all feat dims
        return self.feature_dims[0] * self.feature_dims[1] * self.feature_dims[2]
    
    def _draw_samples(self, z_mu, z_logstd):
        
        z_std = torch.exp(z_logstd)
        z_dim = z_mu.size(1)
        
        # draw samples from variational posterior to calculate
        # E[p(x|z)]
        r = Variable(x.data.new(b,z_dim).normal_())
        z = z_std*r + z_mu
        
        return z
    
    def _decode_pose_to_distribution(self, z, sample_posterior=True):
        # no mu and std dev for class prediction, this is just a class confidence score list of len num_classes
        c_pred = z[..., -self.num_classes:] # torch.Size([8, num_classes])
        # the rest are the moments. first POSE_6D_DIM + LHW_DIM are mu and last POSE_6D_DIM + LHW_DIM are std dev
        bbox_mu = z[..., :POSE_6D_DIM + LHW_DIM] # torch.Size([8, 9])
        bbox_std = z[..., POSE_6D_DIM + LHW_DIM:2*(POSE_6D_DIM + LHW_DIM)] # torch.Size([8, 9])
        bbox_moments = torch.cat([bbox_mu, bbox_std], dim=-1) # torch.Size([8, 18])
        bbox_moments = bbox_moments.to(self.device)
        bbox_posterior = DiagonalGaussianDistribution(bbox_moments)
        return bbox_posterior, c_pred
    
    def _decode_pose(self, x, sample_posterior=True):
        """
        Decode the pose from the given image feature map.
        
        Args:
            x: Input image feature map tensor.
        
        Returns:
            Decoded pose tensor.
        """
        x = x.view(x.size(0), -1)  # flatten the input tensor

        z = self.pose_decoder(x) # torch.Size([8, 19])
        
        bbox_posterior, c_pred = self._decode_pose_to_distribution(z)
        
        if sample_posterior:
            bbox_pred = bbox_posterior.sample()
        else:
            bbox_pred = bbox_posterior.mode()
        
        c = c_pred
        dec_pose = torch.cat([bbox_pred, c], dim=-1) # torch.Size([8, 19])
        return dec_pose, bbox_posterior
    
    def _encode_pose(self, x):
        """
        Encode the pose to get the pose feature map from the given pose tensor.
        
        Args:
            x: Input pose tensor.
        
        Returns:
            Encoded pose feature map tensor.
        """
        # x: torch.Size([4, 8])
        flattened_encoded_pose_feat_map = self.pose_encoder(x) # torch.Size([4, 4096]) = 4, 16*16*16     
        return flattened_encoded_pose_feat_map.view(flattened_encoded_pose_feat_map.size(0), self.feature_dims[0], self.feature_dims[1], self.feature_dims[2])
    
    def encode(self, x):
        x = x.to(self.device) # torch.Size([4, 3, 256, 256])
        h = self.encoder(x) # torch.Size([4, 32, 16, 16])
        moments_obj = self.quant_conv_obj(h) # torch.Size([4, 32, 16, 16])
        pose_feat = self.quant_conv_pose(h) # torch.Size([4, 16, 16, 16])
        posterior_obj = DiagonalGaussianDistribution(moments_obj) # torch.Size([4, 16, 16, 16]) sample
        return posterior_obj, pose_feat
    
    def forward(self, input_im, sample_posterior=True):
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
            # reshape input_im to (batch_size, 3, 256, 256)
            input_im = input_im.to(memory_format=torch.contiguous_format).float() # torch.Size([4, 3, 256, 256])
            posterior_obj, pose_feat = self.encode(input_im) # Distribution: torch.Size([4, 16, 16, 16]), torch.Size([4, 16, 16, 16])
            if sample_posterior: # True
                z_obj = posterior_obj.sample() # torch.Size([4, 16, 16, 16])
            else:
                z_obj = posterior_obj.mode()
            
            # torch.Size([4, 16, 16, 16]), True
            dec_pose, bbox_posterior = self._decode_pose(pose_feat, sample_posterior) # torch.Size([4, 8]), torch.Size([4, 7])
            enc_pose = self._encode_pose(dec_pose) # torch.Size([4, 16, 16, 16])
            
            assert z_obj.shape == enc_pose.shape, f"z_obj shape: {z_obj.shape}, enc_pose shape: {enc_pose.shape}"
            
            z_obj_pose = z_obj + enc_pose # torch.Size([4, 16, 16, 16])
            
            dec_obj = self.decode(z_obj_pose) # torch.Size([4, 3, 256, 256])
            
            return dec_obj, dec_pose, posterior_obj, bbox_posterior
        
    def get_pose_input(self, batch, k):
        x = batch[k] 
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def get_mask_input(self, batch, k):
        if k is None:
            return None
        x = batch[k]
        return x
    
    def get_class_input(self, batch, k):
        x = batch[k]
        return x
    
    def get_bbox_input(self, batch, k):
        x = batch[k]
        return x
    
    def _get_perturbed_pose(self, batch, k):
        x = batch[k]
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        rgb_gt = self.get_input(batch, self.image_rgb_key).permute(0, 2, 1, 3).to(self.device) # torch.Size([4, 3, 256, 256]) 
        pose_gt = self.get_pose_input(batch, self.pose_key).to(self.device) # torch.Size([4, 4]) #
        mask_gt = self.get_mask_input(batch, self.image_mask_key) # None
        mask_gt = mask_gt.to(self.device) if mask_gt is not None else None
        class_gt = self.get_class_input(batch, self.class_key).to(self.device) # torch.Size([4])
        bbox_gt = self.get_bbox_input(batch, self.bbox_key).to(self.device) # torch.Size([4, 3])
        
        # torch.Size([4, 3, 256, 256]), torch.Size([4, 8]), torch.Size([4, 16, 16, 16]), torch.Size([4, 7])
        dec_obj, dec_pose, posterior_obj, bbox_posterior = self.forward(rgb_gt)
        if optimizer_idx == 0:
            # train encoder+decoder+logvar # last layer: torch.Size([3, 128, 3, 3])
            aeloss, log_dict_ae = self.loss(rgb_gt, mask_gt, pose_gt,
                                            dec_obj, dec_pose,
                                            class_gt, bbox_gt,
                                            posterior_obj, bbox_posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(rgb_gt, mask_gt, pose_gt,
                                                dec_obj, dec_pose,
                                                class_gt, bbox_gt,
                                                posterior_obj, bbox_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        
        rgb_gt = self.get_input(batch, self.image_rgb_key).permute(0, 2, 1, 3).to(self.device) # torch.Size([4, 3, 256, 256])
        pose_gt = self.get_pose_input(batch, self.pose_key).to(self.device) # torch.Size([4, 4])
        mask_gt = self.get_mask_input(batch, self.image_mask_key)
        mask_gt = mask_gt.to(self.device) if mask_gt is not None else None
        class_gt = self.get_class_input(batch, self.class_key).to(self.device) # torch.Size([4])
        bbox_gt = self.get_bbox_input(batch, self.bbox_key).to(self.device) # torch.Size([4, 3])
         
        # torch.Size([4, 3, 256, 256]) torch.Size([4, 8]) torch.Size([4, 16, 16, 16]), torch.Size([4, 7])
        dec_obj, dec_pose, posterior_obj, bbox_posterior = self.forward(rgb_gt)
       
        
        _, log_dict_ae = self.loss(rgb_gt, mask_gt, pose_gt,
                                      dec_obj, dec_pose,
                                      class_gt, bbox_gt,
                                      posterior_obj, bbox_posterior, 0, self.global_step,
                                      last_layer=self.get_last_layer(), split="val")

        _, log_dict_disc = self.loss(rgb_gt, mask_gt, pose_gt,
                                        dec_obj, dec_pose,
                                        class_gt, bbox_gt,
                                        posterior_obj, bbox_posterior, 1, self.global_step,
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
    
    def _perturb_poses(self, batch, dec_pose):
        pose_6d_perturbed_v3 = self._get_perturbed_pose(batch, self.pose_perturbed_key).squeeze()[:, -1]
        pose_6d_perturbed_ret = dec_pose.clone() # torch.Size([4, 8])
        pose_6d_perturbed_ret[:, 3] = pose_6d_perturbed_v3 # torch.Size([4, 8])
        assert pose_6d_perturbed_ret.shape == dec_pose.shape, f"pose_6d_perturbed_ret shape: {pose_6d_perturbed_ret.shape}"
        return pose_6d_perturbed_ret.to(self.device) # torch.Size([4, 8])
 
    def _perturbed_pose_forward(self, posterior_obj, dec_pose, batch, sample_posterior=True):
        if sample_posterior:
            z_obj = posterior_obj.sample() # torch.Size([4, 16, 16, 16])
        else:
            z_obj = posterior_obj.mode()
        dec_pose_perturbed = self._perturb_poses(batch, dec_pose) # torch.Size([4, 8])
        enc_pose_perturbed = self._encode_pose(dec_pose_perturbed) # torch.Size([4, 16, 16, 16])
        z_obj_pose_perturbed = z_obj + enc_pose_perturbed # torch.Size([4, 16, 16, 16])
        dec_obj_pose_perturbed = self.decode(z_obj_pose_perturbed) # torch.Size([4, 3, 256, 256])
        return dec_obj_pose_perturbed
            
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        
        x_rgb = self.get_input(batch, self.image_rgb_key).permute(0, 2, 1, 3).to(self.device).float()
        x_mask = self.get_mask_input(batch, self.image_mask_key)
        x_mask = x_mask.to(self.device) if x_mask is not None else None
        
        if x_mask is not None:
            x_mask = x_mask.to(self.device)
            # convert mask to float, 0.0, 1.0 if not already
            if x_mask.dtype != torch.float32:
                x_mask = x_mask.float()
            
        if not only_inputs:
            # torch.Size([4, 3, 256, 256]) torch.Size([4, 8]) torch.Size([4, 16, 16, 16]) torch.Size([4, 7])
            xrec, poserec, posterior_obj, bbox_posterior = self.forward(x_rgb)
            xrec_perturbed_pose = self._perturbed_pose_forward(posterior_obj, poserec, batch) #torch.Size([4, 3, 256, 256])
            
            xrec_rgb = xrec[:, :3, :, :] # torch.Size([8, 3, 64, 64])
            xrec_perturbed_pose_rgb = xrec_perturbed_pose[:, :3, :, :] # torch.Size([4, 3, 256, 256])
            
            if x_rgb.shape[1] > 3:
                # colorize with random projection
                assert xrec_rgb.shape[1] > 3
                x_rgb = self.to_rgb(x_rgb)
                xrec_rgb = self.to_rgb(xrec_rgb)
                xrec_perturbed_pose_rgb = self.to_rgb(xrec_perturbed_pose_rgb)
        
            # scale is 0, 1. scale to -1, 1
            log["reconstructions_rgb"] = torch.tensor(xrec_rgb)
            log["perturbed_pose_reconstruction_rgb"] = torch.tensor(xrec_perturbed_pose_rgb)
        
        log["inputs_rgb"] = x_rgb
        return log
    
    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
