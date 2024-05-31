# src/models/autoencoder.py
import math
import random
from math import radians
import numpy as np
import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torchvision.ops import batched_nms, nms
import pytorch_lightning as pl

from ldm.models.autoencoder import AutoencoderKL
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

from src.modules.autoencodermodules.feat_encoder import FeatEncoder
from src.modules.autoencodermodules.feat_decoder import FeatDecoder
from src.modules.autoencodermodules.pose_encoder import PoseEncoderSpatialVAE as PoseEncoder
from src.modules.autoencodermodules.pose_decoder import PoseDecoderSpatialVAE as PoseDecoder
from src.util.distributions import DiagonalGaussianDistribution
    
from src.data.specs import LABEL_NAME2ID, LABEL_ID2NAME, CAM_NAMESPACE,  POSE_DIM, LHW_DIM, BBOX_3D_DIM, BACKGROUND_CLASS_IDX, BBOX_DIM

try:
    import wandb
except ImportWarning:
    print("WandB not installed")
    wandb = None


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
                 fill_factor_key="fill_factor",
                 pose_perturbed_key="pose_6d_perturbed",
                 class_key="class_id",
                 bbox_key="bbox_sizes",
                 colorize_nlabels=None,
                 monitor=None,
                 activation="relu",
                 feat_dims=[16, 16, 16],
                 pose_decoder_config=None,
                 pose_encoder_config=None,
                 dropout_prob_init=1.0,
                 dropout_prob_final=0.7,
                 dropout_warmup_steps=5000,
                 pose_conditioned_generation_steps=10000,
                 intermediate_img_feature_leak_steps=0,
                 add_noise_to_z_obj=True,
                 train_on_yaw=True,
                 ema_decay=0.999,
                 ):
        pl.LightningModule.__init__(self)
        self.encoder_pretrain_steps = lossconfig["params"]["encoder_pretrain_steps"]
        self.intermediate_img_feature_leak_steps = intermediate_img_feature_leak_steps
        self.train_on_yaw = train_on_yaw
        self.dropout_prob_final = dropout_prob_final # 0.7
        self.dropout_prob_init = dropout_prob_init # 1.0
        self.dropout_prob = self.dropout_prob_init
        self.dropout_warmup_steps = dropout_warmup_steps # 10000 (after stage 1: encoder pretraining)
        self.pose_conditioned_generation_steps = pose_conditioned_generation_steps # 10000
        self.add_noise_to_z_obj = add_noise_to_z_obj
        self.feature_dims = feat_dims
        self.image_rgb_key = image_rgb_key
        self.pose_key = pose_key
        self.pose_perturbed_key = pose_perturbed_key
        self.class_key = class_key
        self.bbox_key = bbox_key
        self.fill_factor_key = fill_factor_key
        self.image_mask_key = image_mask_key
        self.encoder = FeatEncoder(**ddconfig)
        self.decoder = FeatDecoder(**ddconfig)
        lossconfig["params"]["train_on_yaw"] = self.train_on_yaw
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
        
        self.num_classes = lossconfig["params"]["num_classes"]
        enc_feat_dims = self._get_enc_feat_dims(ddconfig)
        
        self.pose_decoder = instantiate_from_config(pose_decoder_config)
        self.pose_encoder = instantiate_from_config(pose_encoder_config)
        self.z_channels = ddconfig["z_channels"]
        self.euler_convention=euler_convention
        self.feat_dims = feat_dims
        
        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
        
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
        bbox_mu = z[..., :POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM] # torch.Size([8, 9])
        bbox_std = z[..., POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM:2*(POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM)] # torch.Size([8, 9])
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
    
    def _get_dropout_prob(self):
        """
        Get the current dropout probability.
        
        Returns:
            Dropout probability.
        """
        if self.global_step < self.encoder_pretrain_steps: # encoder pretraining phase
            # set dropout probability to 1.0
            dropout_prob = self.dropout_prob_init
        elif self.global_step < self.intermediate_img_feature_leak_steps + self.encoder_pretrain_steps:
            dropout_prob = 0.95
        
        elif self.global_step < self.intermediate_img_feature_leak_steps + self.encoder_pretrain_steps + self.pose_conditioned_generation_steps: # pose conditioned generation phase
            dropout_prob = self.dropout_prob_init
        
        elif self.global_step < self.intermediate_img_feature_leak_steps + self.dropout_warmup_steps + self.encoder_pretrain_steps + self.pose_conditioned_generation_steps: # pose conditioned generation phase
            # linearly decrease dropout probability from initial to final value
            if self.dropout_prob_init == self.dropout_prob_final or self.dropout_warmup_steps == 0:
                dropout_prob = self.dropout_prob_final
            else:
                dropout_prob = self.dropout_prob_init - (self.dropout_prob_init - self.dropout_prob_final) * (self.global_step - self.encoder_pretrain_steps) / self.dropout_warmup_steps # 1.0 - (1.0 - 0.7) * (10000 - 10000) / 10000 = 1.0
        
        else: # VAE phase
            # set dropout probability to dropout_prob_final
            dropout_prob = self.dropout_prob_final
        
        return dropout_prob
    
    def forward(self, input_im, sample_posterior=True, second_pose=None):
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
            
            self.dropout_prob = self._get_dropout_prob()
            
            if self.dropout_prob > 0:
                dropout = nn.Dropout(p=self.dropout_prob)
                z_obj = dropout(z_obj)
            
            if self.add_noise_to_z_obj:
                # draw from standard normal distribution
                std_normal = Normal(0, 1)
                z_obj_noise = std_normal.sample(posterior_obj.mean.shape).to(self.device) # torch.Size([4, 16, 16, 16])
                z_obj = z_obj + z_obj_noise
            
            # torch.Size([4, 16, 16, 16]), True
            dec_pose, bbox_posterior = self._decode_pose(pose_feat, sample_posterior) # torch.Size([4, 8]), torch.Size([4, 7])
            # Replace pose with other pose if supervised with other patch
            if second_pose is not None:
                gen_pose = second_pose.to(dec_pose)
            else:
                gen_pose = dec_pose
            
            if self.global_step < self.encoder_pretrain_steps: # no reconstruction loss in this phase
                dec_obj = torch.zeros_like(input_im).to(self.device) # torch.Size([4, 3, 256, 256])
            else:
                enc_pose = self._encode_pose(gen_pose) # torch.Size([4, 16, 16, 16])
                
                assert z_obj.shape == enc_pose.shape, f"z_obj shape: {z_obj.shape}, enc_pose shape: {enc_pose.shape}"
                
                z_obj_pose = z_obj + enc_pose # torch.Size([4, 16, 16, 16])
                
                dec_obj = self.decode(z_obj_pose) # torch.Size([4, 3, 256, 256])
            
            return dec_obj, dec_pose, posterior_obj, bbox_posterior
        
    def get_pose_input(self, batch, k, postfix=""):
        x = batch[k+postfix] 

        if self.train_on_yaw:
            yaw = batch["yaw"+postfix]
            # place yaw at index 3
            x[:, 3] = yaw
        
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
        x = batch[k].squeeze(1) # torch.Size([1, 4])
        if self.train_on_yaw:
            x = torch.zeros_like(x) # torch.Size([4, 6])
            x[:, -1] = batch["yaw_perturbed"] # torch.Size([4])
        return x # torch.Size([1, 4])
    
    def get_fill_factor_input(self, batch, k):
        x = batch[k]
        return x
    
    def get_all_inputs(self, batch, postfix=""):
        # Get RGB GT
        rgb_gt = self.get_input(batch, self.image_rgb_key).permute(0, 2, 3, 1).to(self.device) # torch.Size([4, 3, 256, 256]) 
        rgb_gt = self._rescale(rgb_gt)
        mask_2d_bbox = batch["mask_2d_bbox"]
        # Get Pose GT
        pose_gt = self.get_pose_input(batch, self.pose_key).to(self.device) # torch.Size([4, 4]) #
        # Get 2D Mask
        mask_gt = self.get_mask_input(batch, self.image_mask_key) # None
        mask_gt = mask_gt.to(self.device) if mask_gt is not None else None
        bbox_gt = self.get_bbox_input(batch, self.bbox_key).to(self.device) # torch.Size([4, 3])
        fill_factor_gt = self.get_fill_factor_input(batch, self.fill_factor_key).to(self.device).float()
        # Get Class GT
        class_gt = self.get_class_input(batch, self.class_key).to(self.device) # torch.Size([4])
        class_gt_label = batch["class_name"]

            
        # torch.Size([4, 3, 256, 256]), torch.Size([4, 8]), torch.Size([4, 16, 16, 16]), torch.Size([4, 7])
        if "pose_6d_2" in batch:
            # Replace RGB and Mask with second patch
            rgb_gt = self.get_input(batch, "patch2").permute(0, 2, 3, 1).to(self.device) # torch.Size([4, 3, 256, 256])
            mask_2d_bbox = batch["mask_2d_bbox_2"]
            
            # Get respective pose for forward pass
            snd_pose = self.get_pose_input(batch, self.pose_key, postfix="_2").to(self.device) # torch.Size([4, 4]) #
            snd_bbox = self.get_bbox_input(batch, self.bbox_key).to(self.device) # torch.Size([4, 3]) 
            snd_fill = self.get_fill_factor_input(batch, self.fill_factor_key+"_2").to(self.device).float().unsqueeze(1)

            # snd_pose = snd_pose.unsqueeze(0) if snd_pose.dim() == 1 else snd_pose
            # snd_bbox = snd_bbox.unsqueeze(0) if snd_bbox.dim() == 1 else snd_bbox
            # get one hot encoding for class_id - all classes in self.label_id2class_id.values
            num_classes = self.num_classes
            class_probs = torch.nn.functional.one_hot(torch.tensor(class_gt), num_classes=num_classes).float()
            second_pose = torch.cat((snd_pose, snd_bbox, snd_fill, class_probs), dim=1)
        else:
            second_pose = None
            
        return rgb_gt, pose_gt, mask_gt, class_gt, class_gt_label, bbox_gt, fill_factor_gt, mask_2d_bbox, second_pose
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # Get inputs in right shape
        rgb_gt, pose_gt, mask_gt, class_gt, class_gt_label, bbox_gt, fill_factor_gt, mask_2d_bbox, second_pose = self.get_all_inputs(batch)
        # Run full forward pass
        
        #### PLEASE DEBUG 2D MASK ####
        mask_2d_bbox = torch.ones_like(mask_2d_bbox)
        #### PLEASE DEBUG 2D MASK ####
        
        dec_obj, dec_pose, posterior_obj, bbox_posterior = self.forward(rgb_gt, second_pose=second_pose)
        self.log("dropout_prob", self.dropout_prob, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # Train the autoencoder
        if optimizer_idx == 0:
            # train encoder+decoder+logvar # last layer: torch.Size([3, 128, 3, 3])
            aeloss, log_dict_ae = self.loss(rgb_gt, mask_gt, pose_gt,
                                            dec_obj, dec_pose,
                                            class_gt, class_gt_label, bbox_gt, fill_factor_gt,
                                            posterior_obj, bbox_posterior, optimizer_idx, self.global_step, mask_2d_bbox,
                                            last_layer=self.get_last_layer(), split="train")            
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        # Train the discriminator
        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(rgb_gt, mask_gt, pose_gt,
                                                dec_obj, dec_pose,
                                                class_gt, class_gt_label, bbox_gt, fill_factor_gt,
                                                posterior_obj, bbox_posterior, optimizer_idx, self.global_step, mask_2d_bbox,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        
        rgb_gt = self.get_input(batch, self.image_rgb_key).permute(0, 2, 3, 1).to(self.device) # torch.Size([4, 3, 256, 256])
        rgb_gt = self._rescale(rgb_gt)
        pose_gt = self.get_pose_input(batch, self.pose_key).to(self.device) # torch.Size([4, 4])
        mask_gt = self.get_mask_input(batch, self.image_mask_key)
        mask_gt = mask_gt.to(self.device) if mask_gt is not None else None
        class_gt = self.get_class_input(batch, self.class_key).to(self.device) # torch.Size([4])
        class_gt_label = batch["class_name"]
        bbox_gt = self.get_bbox_input(batch, self.bbox_key).to(self.device) # torch.Size([4, 3])
        fill_factor_gt = self.get_fill_factor_input(batch, self.fill_factor_key).to(self.device).float() 
        # torch.Size([4, 3, 256, 256]) torch.Size([4, 8]) torch.Size([4, 16, 16, 16]), torch.Size([4, 7])
        dec_obj, dec_pose, posterior_obj, bbox_posterior = self.forward(rgb_gt)
        mask_2d_bbox = batch["mask_2d_bbox"]
        
        _, log_dict_ae = self.loss(rgb_gt, mask_gt, pose_gt,
                                      dec_obj, dec_pose,
                                      class_gt, class_gt_label, bbox_gt,fill_factor_gt,
                                      posterior_obj, bbox_posterior, 0, 
                                      global_step=self.global_step, mask_2d_bbox=mask_2d_bbox,
                                      last_layer=self.get_last_layer(), split="val")

        _, log_dict_disc = self.loss(rgb_gt, mask_gt, pose_gt,
                                        dec_obj, dec_pose,
                                        class_gt, class_gt_label, bbox_gt,fill_factor_gt,
                                        posterior_obj, bbox_posterior, 1, 
                                        global_step=self.global_step, mask_2d_bbox=mask_2d_bbox,
                                        last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def test_step(self, batch, batch_idx):
        # TODO: Move to init
        self.chunk_size = 128
        self.class_thresh = 0.3 # TODO: Set to 0.5 or other value that works on val set
        self.fill_factor_thresh = 0.5 # TODO: Set to 0.5 or other value that works on val set
        self.num_refinement_steps = 10
        self.ref_lr=1.0e0
        
        # Prepare Input
        input_patches = batch[self.image_rgb_key].to(self.device) # torch.Size([B, 3, 256, 256])
        assert input_patches.dim() == 4 or (input_patches.dim() == 5 and input_patches.shape[0] == 1), f"Only supporting batch size 1. Input_patches shape: {input_patches.shape}"
        if input_patches.dim() == 5:
            input_patches = input_patches.squeeze(0) # torch.Size([B, 3, 256, 256])
        input_patches = self._rescale(input_patches) # torch.Size([B, 3, 256, 256])
        
        # Chunked dec_pose[..., :POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM]
        all_objects = []
        all_poses = []
        all_patch_indeces = []
        all_scores = []
        all_classes = []
        chunk_size = self.chunk_size
        with torch.no_grad():
            for i in range(0, len(input_patches), chunk_size):
                selected_patches = input_patches[i:i+chunk_size]
                global_patch_index = i + torch.arange(chunk_size)[:len(selected_patches)]
                selected_patch_indeces, z_obj, dec_pose, score, class_idx = self._get_valid_patches(selected_patches, global_patch_index)
                all_patch_indeces.append(selected_patch_indeces)
                all_objects.append(z_obj)
                all_poses.append(dec_pose)
                all_scores.append(score)
                all_classes.append(class_idx)
                
        scores = torch.cat(all_scores)
                    
        # Inference refinement
        all_patch_indeces = torch.cat(all_patch_indeces)
        if not len(all_patch_indeces):
            return torch.empty(0)
        all_z_objects = torch.cat(all_objects)
        all_z_poses = torch.cat(all_poses)
        patches_w_objs = input_patches[all_patch_indeces]
        dec_pose_refined = self._refinement_step(patches_w_objs, all_z_objects, all_z_poses)
        
        # TODO: save everything to an output file!
        return dec_pose_refined
    
    def _get_valid_patches(self, input_patches, global_patch_index):
        local_patch_idx = torch.arange(len(global_patch_index))
        # Run encoder
        posterior_obj, pose_feat = self.encode(input_patches)
        z_obj = posterior_obj.mode()
        dec_pose, bbox_posterior  = self._decode_pose(pose_feat, sample_posterior=False)
        
        # Threshold by class probabilities
        # TODO: Check class probabilities after sofmax
        class_pred = dec_pose[:, -self.num_classes:]
        class_prob = torch.softmax(class_pred, dim=-1)
        # obj_prob = class_prob[:, -self.num_classes:]
        score, class_idx = torch.max(class_prob, -1)
        class_thresh_mask = ((score > self.class_thresh) # Exclude unlikely patches
                                # & (bckg_prob < self.class_thresh).squeeze(-1) 
                                & ~(class_idx == self.num_classes-1) # Exclude background class
                                )
        if not class_thresh_mask.sum():
            return self.return_empty(global_patch_index, z_obj, dec_pose, score, class_idx)
        local_patch_idx = local_patch_idx[class_thresh_mask]
        
        # Threshold by fill factor
        fill_factor = dec_pose[:, POSE_6D_DIM + LHW_DIM : POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM][local_patch_idx].squeeze(-1)
        fill_factor_mask = fill_factor > self.fill_factor_thresh # TODO: Check negative or positive fill factor
        if not fill_factor_mask.sum():
            return self.return_empty(global_patch_index, z_obj, dec_pose, score, class_idx)
        local_patch_idx = local_patch_idx[fill_factor_mask]
        
        # TODO: Finish when transformations are available
        # # NMS on the BEV plane
        # bbox_enc = dec_pose[:, POSE_6D_DIM + LHW_DIM][local_patch_idx]
        # # TODO: Transform the box to 3D
        # bbox3D = transform_to_3D(bbox_enc)
        # # TODO: Only WL and XY on BEV + reshape to (N, 4) box
        # bbox_BEV = transform_to_BEV(bbox3D)
        # class_scores = class_prob[local_patch_idx]
        # class_prediction = torch.argmax(class_scores, dim=-1)
        
        # nms_box_mask = batched_nms(boxes=bbox_BEV, scores=class_scores, idxs=class_prediction, iou_threshold=0.5)
        # local_patch_idx = local_patch_idx[nms_box_mask]       
        
        
        return global_patch_index[local_patch_idx], z_obj[local_patch_idx], dec_pose[local_patch_idx], score[local_patch_idx], class_idx[local_patch_idx]
    
    def return_empty(self, global_patch_index, z_obj, dec_pose, score, class_idx):
        return torch.empty_like(global_patch_index[:0]), torch.empty_like(z_obj[:0]), torch.empty_like(dec_pose[:0]), torch.empty_like(score[:0]), torch.empty_like(class_idx[:0])
  
    @torch.enable_grad()
    def _refinement_step(self, input_patches, z_obj, z_pose):
        # Initialize optimizer and parameters
        refined_pose = z_pose[:, :-self.num_classes]
        obj_class = z_pose[:, -self.num_classes:]
        refined_pose_param = nn.Parameter(refined_pose, requires_grad=True)
        optim_refined = self._init_refinement_optimizer(refined_pose_param, lr=self.ref_lr)
        # Run K iter refinement steps
        for k in range(self.num_refinement_steps):
            dec_pose = torch.cat([refined_pose_param, obj_class], dim=-1)
            optim_refined.zero_grad()
            enc_pose = self._encode_pose(dec_pose)
            z_obj_pose = z_obj + enc_pose
            gen_image = self.decode(z_obj_pose)
            rec_loss = self.loss._get_rec_loss(input_patches, gen_image, use_pixel_loss=True).mean()
            rec_loss.backward()
            optim_refined.step()
            print("Gradient: ", refined_pose_param.grad.mean(), refined_pose_param.grad.std())
            print("Poses: ", refined_pose_param.mean(), refined_pose_param.std())
        
        dec_pose = torch.cat([refined_pose, obj_class], dim=-1)   
        return dec_pose.data
    
    def _init_refinement_optimizer(self, pose, lr=1e-3):
        return torch.optim.Adam([pose], lr=lr)
    
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
        pose_6d_perturbed_v3 = self._get_perturbed_pose(batch, self.pose_perturbed_key)[:, -1]
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
        
        x_rgb = self.get_input(batch, self.image_rgb_key).permute(0, 2, 3, 1).to(self.device).float()
        x_rgb = self._rescale(x_rgb)
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
            log["reconstructions_rgb"] = xrec_rgb.clone().detach()
            log["perturbed_pose_reconstruction_rgb"] = xrec_perturbed_pose_rgb.clone().detach()
        
        log["inputs_rgb"] = x_rgb.clone().detach()
        return log
    
    def _rescale(self, x):
        # scale is -1
        return 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    
    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
