import torch
import numpy as np
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution as LDM_DiagonalGaussianDistribution

class DiagonalGaussianDistribution(LDM_DiagonalGaussianDistribution):
    def __init__(self, parameters, deterministic=False):
        super(DiagonalGaussianDistribution, self).__init__(parameters, deterministic)
        self.device = parameters.device
        
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                # make sure all tensors are on the same device
                self.mean = self.mean.to(self.device)
                self.var = self.var.to(self.device)
                self.logvar = self.logvar.to(self.device)
                other.mean = other.mean.to(self.device)
                other.var = other.var.to(self.device)
                other.logvar = other.logvar.to(self.device)
                
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])