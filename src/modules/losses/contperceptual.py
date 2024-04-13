# src/modules/losses/contperceptual.py
import torch.nn as nn
from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator as LPIPSWithDiscriminator_LDM
from taming.modules.losses.vqperceptual import adopt_weight
import torch

class LPIPSWithDiscriminator(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class PoseLoss(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, pose_weight=1.0, mask_weight=1.0, pose_loss_fn=None, 
                 mask_loss_fn=None, use_mask_loss=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_weight = pose_weight
        self.mask_weight = mask_weight
        self.use_mask_loss = use_mask_loss
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
        
    def compute_pose_loss(self, pred, gt):
        pose_loss = self.pose_loss(pred, gt)
        weighted_pose_loss = self.pose_weight * pose_loss
        return pose_loss, weighted_pose_loss

    def _get_rec_loss(self, inputs, reconstructions):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        return rec_loss

    def _get_nll_loss(self, rec_loss, weights=None):
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
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
    
    def forward(self, 
                rgb_gt, mask_gt, pose_gt,
                dec_obj, dec_pose,
                posterior_obj, posterior_pose, optimizer_idx, global_step, 
                last_layer=None, cond=None, split="train",
                weights=None):
        assert pose_inputs1.shape == pose_reconstructions1.shape, \
            f"pose_inputs.shape: {pose_inputs1.shape}, \
                pose_reconstructions.shape: {pose_reconstructions1.shape}"

        gt_obj = torch.cat((rgb_gt, mask_gt), dim=1)
        inputs, reconstructions = gt_obj, dec_obj
        
        inputs_rgb = inputs[:, :3, :, :]
        inputs_mask = inputs[:, 3:, :, :]
        
        reconstructions_rgb = reconstructions[:, :3, :, :]
        
        # if reconstructions have alpha channel, store it in mask
        
        if reconstructions.shape[1] == 4:
            reconstructions_mask = reconstructions[:, 3:, :, :]
        else:
            reconstructions_mask = torch.zeros_like(inputs_mask)
            self.use_mask_loss = False

        pose_loss, weighted_pose_loss = self.compute_pose_loss(pose_gt, dec_pose)
        mask_loss, weighted_mask_loss = self.get_mask_loss(inputs_mask, reconstructions_mask)
        
        rec_loss = self._get_rec_loss(inputs_rgb, reconstructions_rgb)
        nll_loss, weighted_nll_loss = self._get_nll_loss(rec_loss, weights)
        
        kl_loss_obj = self._get_kl_loss(posterior_obj)
        kl_loss_pose = self._get_kl_loss(posterior_pose)
        kl_loss = kl_loss_obj + kl_loss_pose

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
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
                + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss_obj".format(split): kl_loss_obj.detach().mean(),
                   "{}/kl_loss_pose".format(split): kl_loss_pose.detach().mean(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), 
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/weighted_nll_loss".format(split): weighted_nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/pose_loss".format(split): pose_loss.detach().mean(),
                   "{}/weighted_pose_loss".format(split): weighted_pose_loss.detach().mean(),
                   "{}/mask_loss".format(split): mask_loss.detach().mean(),
                   "{}/weighted_mask_loss".format(split): weighted_mask_loss.detach().mean()
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
