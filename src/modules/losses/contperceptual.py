# src/modules/losses/contperceptual.py
import torch.nn as nn
from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator as LPIPSWithDiscriminator_LDM
from taming.modules.losses.vqperceptual import adopt_weight
import torch
import math
import logging

SE3_DIM = 16

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
    
    def _get_combined_inputs_reconstructions(self, inputs1, inputs2, reconstructions1, reconstructions2, optimizer_idx, global_step):
        """
        Get the combined inputs and reconstructions based on the global step and optimizer index.

        Args:
            inputs1: The first set of inputs.
            inputs2: The second set of inputs.
            reconstructions1: The first set of reconstructions.
            reconstructions2: The second set of reconstructions.
            optimizer_idx: The index of the optimizer.
            global_step: The global step.

        Returns:
            inputs: The selected inputs.
            reconstructions: The selected reconstructions.
        """
        if global_step % 2 == 0:
            logging.info(f"Using inputs1 and reconstructions1 for global step {global_step} and optimizer_idx {optimizer_idx}")
            inputs = inputs1
            reconstructions = reconstructions1
        else:
            logging.info(f"Using inputs2 and reconstructions2 for global step {global_step} and optimizer_idx {optimizer_idx}")
            inputs = inputs2
            reconstructions = reconstructions2
        return inputs, reconstructions
    
    def forward(self, inputs1, inputs2, 
                reconstructions1, reconstructions2, 
                pose_inputs1, pose_reconstructions1,
                posteriors1, optimizer_idx, global_step, 
                last_layer=None, cond=None, split="train",
                weights=None):
        assert pose_inputs1.shape == pose_reconstructions1.shape, \
            f"pose_inputs.shape: {pose_inputs1.shape}, pose_reconstructions.shape: {pose_reconstructions1.shape}"
        assert pose_inputs1.shape[1] == int(math.sqrt(SE3_DIM))
        assert pose_inputs1.shape[2] == int(math.sqrt(SE3_DIM))

        inputs, reconstructions = self._get_combined_inputs_reconstructions(inputs1, inputs2, reconstructions1, reconstructions2, optimizer_idx, global_step)
        
        pose_loss = self.compute_pose_loss(pose_inputs1, pose_reconstructions1)
        weighted_pose_loss = self.pose_weight * pose_loss
        
        rec_loss = self._get_rec_loss(inputs, reconstructions)
        
        nll_loss, weighted_nll_loss = self._get_nll_loss(rec_loss, weights)
        
        kl_loss = self._get_kl_loss(posteriors1)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_pose_loss + weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), 
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/weighted_nll_loss".format(split): weighted_nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/pose_loss".format(split): pose_loss.detach().mean(),
                   "{}/weighted_pose_loss".format(split): weighted_pose_loss.detach().mean()
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
