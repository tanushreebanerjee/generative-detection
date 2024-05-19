import torch
import numpy as np
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution as LDM_DiagonalGaussianDistribution

class DiagonalGaussianDistribution(LDM_DiagonalGaussianDistribution):
    def __init__(self, parameters, deterministic=False):
        super(DiagonalGaussianDistribution, self).__init__(parameters, deterministic)
        # self.device = parameters.device
        
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
                # other_device = other.mean.device
                # self.mean = self.mean.to(other_device)
                # self.var = self.var.to(other_device)
                # self.logvar = self.logvar.to(other_device)
                # other.mean = other.mean.to(other_device)
                # other.var = other.var.to(other_device)
                # other.logvar = other.logvar.to(other_device)
                
                # adding batch dim
                other_mean = other.mean.squeeze().unsqueeze(0).to(self.mean) # other.mean torch.Size([1, 9])
                other_var = other.var.squeeze().unsqueeze(0).to(self.mean)
                other_logvar = other.logvar.squeeze().unsqueeze(0).to(self.mean)
                
              
                num_dims = len(other_mean.size())
                sum_dim = list(range(1, num_dims)) # [1]
                
                # other_logvar torch.Size([1, 9])
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other_mean, 2) / (other_var + 1e-5)
                    + self.var / (other_var + 1e-5) - 1.0 - self.logvar + other_logvar,
                    dim=sum_dim)
                