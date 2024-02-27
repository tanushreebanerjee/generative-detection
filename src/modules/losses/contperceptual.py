# src/modules/losses/contperceptual.py
import torch.nn as nn
from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator as LPIPSWithDiscriminator_LDM

class LPIPSWithDiscriminator(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class PoseLoss(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, pose_weight=1.0, class_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_weight = pose_weight
        self.class_weight = class_weight

        self.pose_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

    def gt_pose_to_so3(self, gt):
        raise NotImplementedError
    
    def compute_pose_loss(self, pred, gt):
        gt = self.gt_pose_to_so3(gt)
        return self.pose_loss(pred, gt)
    
    def compute_class_loss(self, pred, gt):
        return self.class_loss(pred, gt)
    
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        
        loss, log \
            = super().forward(inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer, cond, split,
                weights)
    
        pose_loss = self.compute_pose_loss(reconstructions, inputs)
        class_loss = self.compute_class_loss(posteriors, inputs)
        
        weighted_pose_loss = self.pose_weight * pose_loss
        weighted_class_loss = self.class_weight * class_loss
        
        loss += weighted_pose_loss + weighted_class_loss
        
        log["{}/pose_loss".format(split)] = pose_loss.clone().detach().mean()
        log["{}/class_loss".format(split)] = class_loss.clone().detach().mean()
        log["{}/weighted_pose_loss".format(split)] = weighted_pose_loss.clone().detach().mean()
        log["{}/weighted_class_loss".format(split)] = weighted_class_loss.clone().detach().mean()
        log["{}/total_loss".format(split)] = loss.clone().detach().mean()
        
        return loss, log
            