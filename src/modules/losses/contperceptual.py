# src/modules/losses/contperceptual.py
import torch.nn as nn
from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator as LPIPSWithDiscriminator_LDM
from se3.homtrans3d import T2xyzrpy
import logging

class LPIPSWithDiscriminator(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class PoseLoss(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, pose_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_weight = pose_weight
        self.pose_loss = nn.MSELoss()

    def compute_pose_loss(self, pred, gt):
        return self.pose_loss(pred, gt)
            
    def forward(self, inputs, reconstructions, pose_input, pose_decoded,
                posteriors, optimizer_idx, global_step, 
                last_layer=None, cond=None, split="train",
                weights=None):
        
        loss, log \
            = super().forward(inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer, cond, split,
                weights)

        x_in, y_in, z_in, roll_in, pitch_in, yaw_in = T2xyzrpy(pose_input)
        x_dec, y_dec, z_dec, roll_dec, pitch_dec, yaw_dec = T2xyzrpy(pose_decoded)
        logging.info("pose_input (xyzrpy): %f %f %f %f %f %f", x_in, y_in, z_in, roll_in, pitch_in, yaw_in)
        logging.info("pose_decoded (xyzrpy): %f %f %f %f %f %f", x_dec, y_dec, z_dec, roll_dec, pitch_dec, yaw_dec)
        pose_loss = self.compute_pose_loss(pose_input, pose_decoded)
        weighted_pose_loss = self.pose_weight * pose_loss
           
        loss += weighted_pose_loss 
        
        log["{}/pose_loss".format(split)] = pose_loss.clone().detach().mean()
        log["{}/weighted_pose_loss".format(split)] = weighted_pose_loss.clone().detach().mean()
        log["{}/total_loss".format(split)] = loss.clone().detach().mean()
        
        return loss, log
            