# src/models/autoencoder.py

from ldm.models.autoencoder import AutoencoderKL
from src.modules.autoencodermodules.feat_encoder import FeatEncoder
from src.modules.autoencodermodules.feat_decoder import FeatDecoder
from src.modules.autoencodermodules.pose_encoder import PoseEncoder
from src.modules.autoencodermodules.pose_decoder import PoseDecoder
from ldm.util import instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from se3.homtrans3d import xyzrpy2T, T2xyzrpy
import torch
import pytorch_lightning as pl
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
                 pose_key="object_pose",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        pl.LightningModule.__init__(self)
        self.image_key = image_key
        self.pose_key = pose_key
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
                                                    int(math.sqrt(flattened_encoded_pose_feat_map.size(1)/self.z_channels)), 
                                                    int(math.sqrt(flattened_encoded_pose_feat_map.size(1)/self.z_channels)))
    
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
            pose_decoded_reshaped = pose_decoded.reshape(pose_decoded.size(0), int(math.sqrt(SE3_DIM)), int(math.sqrt(SE3_DIM)))
            return feat_map_img_pose, pose_decoded_reshaped
    
    def forward(self, input, sample_posterior=True):
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
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        z, pose_decoded = self._get_feat_map_img_pose(z)
            
        dec = self.decode(z)

        return dec, posterior, pose_decoded
        
    def get_pose_input(self, batch, k):
        x = batch[k] 
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        pose_inputs = self.get_pose_input(batch, self.pose_key)
        
        reconstructions, posterior, pose_reconstructions = self(inputs)
        
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, pose_inputs, pose_reconstructions,
                                            posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, pose_inputs, pose_reconstructions,
                                                posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        pose_inputs = self.get_pose_input(batch, self.pose_key)
        reconstructions, posterior, pose_reconstructions = self(inputs)
        _, log_dict_ae = self.loss(inputs, reconstructions, pose_inputs, pose_reconstructions,
                                        posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        _, log_dict_disc = self.loss(inputs, reconstructions, pose_inputs, pose_reconstructions,
                                            posterior, 1, self.global_step,
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
    
    def _perturb_angle(self, angle, perturb_angle=15):
        return math.radians(math.degrees(angle) + perturb_angle)
    
    def _perturb_poses(self, pose_inputs, perturb_dims=["p", "y"], perturb_angle=15):
        x, y, z, r, p, y = T2xyzrpy(pose_inputs)
        if "p" in perturb_dims:
            p = self._perturb_angle(p, perturb_angle)
        if "y" in perturb_dims:
            y = self._perturb_angle(y, perturb_angle)
        T_perturbed = torch.tensor(xyzrpy2T(x, y, z, r, p, y), dtype=torch.float32)
        return T_perturbed

    def _get_feat_map_img_perturbed_pose(self, img_feat_map, pose_decoded_perturbed):
            pose_feat_map = self._encode_pose(pose_decoded_perturbed)         
            feat_map_img_pose = img_feat_map + pose_feat_map
            return feat_map_img_pose
    
    def _perturbed_pose_forward(self, posterior, pose_decoded, sample_posterior=True):
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        pose_decoded_perturbed = self._perturb_poses(pose_decoded)
        z = self._get_feat_map_img_perturbed_pose(z, pose_decoded_perturbed)
        dec = self.decode(z)
        return dec
            
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior, pose_decoded = self(x)
            xrec_perturbed_pose = self._perturbed_pose_forward(posterior, pose_decoded)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
            log["perturbed_pose_reconstructions"] = xrec_perturbed_pose
        log["inputs"] = x
        return log
