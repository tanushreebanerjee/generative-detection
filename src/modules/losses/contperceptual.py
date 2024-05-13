# src/modules/losses/contperceptual.py
import torch.nn as nn
from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator as LPIPSWithDiscriminator_LDM
from src.util.distributions import DiagonalGaussianDistribution
from taming.modules.losses.vqperceptual import adopt_weight
import torch
import json
import pickle as pkl
import math
from mmdet.models.losses.focal_loss import FocalLoss

POSE_6D_DIM = 4
LHW_DIM = 3
PROB_THRESHOLD_OBJ = 0.2

class LPIPSWithDiscriminator(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class PoseLoss(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, kl_weight_obj=1.0, kl_weight_bbox=1e-6, pose_weight=1.0, mask_weight=0.0, class_weight=1.0, bbox_weight=1.0,
                 pose_loss_fn=None, mask_loss_fn=None, 
                 use_mask_loss=True, use_class_loss=False, use_bbox_loss=False,
                 num_classes=1, dataset_stats_path="dataset_stats/combined.pkl",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_weight = pose_weight
        self.mask_weight = mask_weight
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.use_mask_loss = use_mask_loss
        self.num_classes = num_classes
        self.kl_weight_obj = kl_weight_obj
        self.kl_weight_bbox = kl_weight_bbox
        assert pose_loss_fn is not None, "Please provide a pose loss function."
        assert mask_loss_fn is not None, "Please provide a mask loss function."
        assert pose_loss_fn in ["l1", "l2", "mse"], \
            "Please provide a valid pose loss function in ['l1', 'l2', 'mse']."
        
        if pose_loss_fn == "l1":
            self.pose_loss = nn.L1Loss()
        elif pose_loss_fn == "l2" or pose_loss_fn == "mse":
            self.pose_loss = nn.MSELoss()
        else:
            raise ValueError("Invalid pose loss function. Please provide a valid pose loss function in ['l1', 'l2', 'mse'].")
        
        if mask_loss_fn == "l1":
            self.mask_loss = nn.L1Loss()
        elif mask_loss_fn == "l2" or mask_loss_fn == "mse":
            self.mask_loss = nn.MSELoss()
        else:
            raise ValueError("Invalid mask loss function. Please provide a valid mask loss function in ['l1', 'l2', 'mse'].")
        
        self.class_loss_fn = FocalLoss() #nn.CrossEntropyLoss()
        self.bbox_loss_fn = nn.MSELoss()
        
        # TODO: need to test this + store in the expected format
        with open(dataset_stats_path, "rb") as handle:
            dataset_stats = pkl.load(handle)
        
        print("Loaded dataset stats:")
        print(dataset_stats)
        self.bbox_distribution = self._create_distribution_from_dataset_stats(dataset_stats)
            
    def _create_distribution_from_dataset_stats(self, dataset_stats):
        bbox_means = torch.zeros(POSE_6D_DIM + LHW_DIM)
        bbox_logvars = torch.zeros(POSE_6D_DIM + LHW_DIM)
        for idx, key in enumerate(["t1", "t2", "t3", "v3", "l", "h", "w"]):
            if key not in dataset_stats: 
                mean = 0.0
                std_dev = 1.0
                logvar = 2. * math.log(std_dev)
                bbox_means[idx] = mean
                bbox_logvars[idx] = logvar
            else: # t1, t2, t3, l, h, w
                mean, logvar = dataset_stats[key]
                bbox_means[idx] = mean
                bbox_logvars[idx] = logvar
                
        parameters = torch.cat((bbox_means.unsqueeze(1), bbox_logvars.unsqueeze(1)), dim=1)
        parameters = torch.tensor(parameters, device="cuda" if torch.cuda.is_available() else "cpu")
        return DiagonalGaussianDistribution(parameters)
       
    def compute_pose_loss(self, pred, gt):
        # need to get loss split for each part of the pose - t1, t2, t3, v3
        assert pred.shape == gt.shape, "Prediction and ground truth shapes do not match."
        assert pred.shape[1] == POSE_6D_DIM, "Invalid pose dimensionality."
        
        t1_loss = self.pose_loss(pred[:, 0], gt[:, 0])
        t2_loss = self.pose_loss(pred[:, 1], gt[:, 1])
        t3_loss = self.pose_loss(pred[:, 2], gt[:, 2])
        v3_loss = self.pose_loss(pred[:, 3], gt[:, 3])
        
        pose_loss = t1_loss + t2_loss + t3_loss + v3_loss
        weighted_pose_loss = self.pose_weight * pose_loss
        
        return pose_loss, weighted_pose_loss, t1_loss, t2_loss, t3_loss, v3_loss

    def _get_rec_loss(self, inputs, reconstructions):  
        # MSE RGB
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) # torch.Size([4, 3, 256, 256])
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()) # torch.Size([4, 1, 1, 1])
            rec_loss = rec_loss + self.perceptual_weight * p_loss # torch.Size([4, 3, 256, 256])
        return rec_loss # torch.Size([4, 3, 256, 256])

    def _get_nll_loss(self, rec_loss, weights=None):
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar # torch.Size([4, 3, 256, 256])
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        return nll_loss, weighted_nll_loss
    
    def _get_kl_loss(self, posteriors):
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        return kl_loss
    
    def get_mask_loss(self, mask1, mask2):
        if self.use_mask_loss:
            mask_loss = self.mask_loss(mask1, mask2)
            weighted_mask_loss = self.mask_weight * mask_loss
        else:
            mask_loss = torch.tensor(0.0)
            weighted_mask_loss = torch.tensor(0.0)
        return mask_loss, weighted_mask_loss
    
    def compute_class_loss(self, class_gt, class_probs):
        # class_gt: torch.Size([4])
        # class_probs: torch.Size([4, 1])
        class_loss = self.class_loss_fn(class_probs, class_gt)
        weighted_class_loss = self.class_weight * class_loss
        return class_loss, weighted_class_loss
    
    def compute_bbox_loss(self, bbox_gt, bbox_pred):
        bbox_loss = nn.MSELoss()(bbox_gt, bbox_pred)
        weighted_bbox_loss = self.bbox_weight * bbox_loss
        return bbox_loss, weighted_bbox_loss
    
    def compute_pose_kl_loss(self, bbox_posterior):
        pose_kl_loss = bbox_posterior.kl(self.bbox_distribution) # torch.Size([4])
        pose_kl_loss = torch.sum(pose_kl_loss) / pose_kl_loss.shape[0] # mean over batch
        return pose_kl_loss
      
    def forward(self, 
                rgb_gt, mask_gt, pose_gt,
                dec_obj, dec_pose,
                class_gt, bbox_gt,
                posterior_obj, bbox_posterior, optimizer_idx, global_step, 
                last_layer=None, cond=None, split="train",
                weights=None):
        
        # rgb_gt: torch.Size([4, 3, 256, 256])
        # pose_gt: torch.Size([4, 4])
        # mask_gt: torch.Size([4, 1, 256, 256]) / None
        # dec_obj: torch.Size([4, 3, 256, 256])
        # dec_pose: torch.Size([4, 8])
        # class_gt: torch.Size([4])
        # 
        if mask_gt == None: # True
            mask_gt = torch.zeros_like(rgb_gt[:, :1, :, :]) # torch.Size([4, 1, 256, 256])
            self.use_mask_loss = False
            gt_obj = rgb_gt # torch.Size([4, 3, 256, 256])
        else:
            gt_obj = torch.cat((rgb_gt, mask_gt), dim=1)
        inputs, reconstructions = gt_obj, dec_obj # torch.Size([4, 3, 256, 256]), torch.Size([4, 3, 256, 256])
        
        inputs_rgb = rgb_gt # torch.Size([4, 3, 256, 256])
        inputs_mask = mask_gt # torch.Size([4, 1, 256, 256])
        

        reconstructions_rgb = reconstructions[:, :3, :, :] # torch.Size([4, 3, 256, 256])
        
        # if reconstructions have alpha channel, store it in mask
        
        if reconstructions.shape[1] == 4:
            reconstructions_mask = reconstructions[:, 3:, :, :]
        else:
            reconstructions_mask = torch.zeros_like(inputs_mask) # torch.Size([4, 1, 256, 256])
            self.use_mask_loss = False
        
        # first POSE_6D_DIM in pose_rec are the 6D pose, next 3 are the LHW and rest is class probs
        pose_rec = dec_pose[:, :POSE_6D_DIM] # torch.Size([4, 4])
       
        lhw_rec = dec_pose[:, POSE_6D_DIM:POSE_6D_DIM+LHW_DIM] # torch.Size([4, 3])
        class_probs = dec_pose[:, POSE_6D_DIM+LHW_DIM:] # torch.Size([4, 1])
            
        class_loss, weighted_class_loss = self.compute_class_loss(class_gt, class_probs)
        bbox_loss, weighted_bbox_loss = self.compute_bbox_loss(bbox_gt, lhw_rec)
        
        pose_loss, weighted_pose_loss, t1_loss, t2_loss, t3_loss, v3_loss = self.compute_pose_loss(pose_gt, pose_rec)
        mask_loss, weighted_mask_loss = self.get_mask_loss(inputs_mask, reconstructions_mask)
        
        rec_loss = self._get_rec_loss(inputs_rgb, reconstructions_rgb)
        nll_loss, weighted_nll_loss = self._get_nll_loss(rec_loss, weights)
        
        kl_loss_obj = self._get_kl_loss(posterior_obj)
        
        kl_loss_obj_bbox = self.compute_pose_kl_loss(bbox_posterior)
    
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous()) # torch.Size([4, 1, 30, 30])
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, 
                                       threshold=self.discriminator_iter_start)
            loss = weighted_pose_loss + weighted_mask_loss + weighted_nll_loss \
                + (self.kl_weight_obj * kl_loss_obj) + (self.kl_weight_bbox * kl_loss_obj_bbox) \
                    + d_weight * disc_factor * g_loss

            log =  {"{}/total_loss".format(split): loss.clone().detach().mean(), 
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/kl_loss_obj".format(split): kl_loss_obj.detach().mean(), 
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/weighted_nll_loss".format(split): weighted_nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                    "{}/pose_loss".format(split): pose_loss.detach().mean(),
                    "{}/weighted_pose_loss".format(split): weighted_pose_loss.detach().mean(),
                    "{}/mask_loss".format(split): mask_loss.detach().mean(),
                    "{}/weighted_mask_loss".format(split): weighted_mask_loss.detach().mean(),
                    "{}/class_loss".format(split): class_loss.detach(),
                    "{}/weighted_class_loss".format(split): weighted_class_loss.detach(),
                    "{}/bbox_loss".format(split): bbox_loss.detach(),
                    "{}/weighted_bbox_loss".format(split): weighted_bbox_loss.detach(),
                    "{}/t1_loss".format(split): t1_loss.detach().mean(),
                    "{}/t2_loss".format(split): t2_loss.detach().mean(),
                    "{}/t3_loss".format(split): t3_loss.detach().mean(),
                    "{}/v3_loss".format(split): v3_loss.detach().mean(),
                    "{}/kl_loss_bbox".format(split): kl_loss_obj_bbox.detach().mean(),
                    "{}/weighted_kl_loss_bbox".format(split): self.kl_weight_bbox * kl_loss_obj_bbox.detach().mean(),
                    "{}/weighted_kl_loss_obj".format(split): self.kl_weight_obj * kl_loss_obj.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, 
                                       threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
