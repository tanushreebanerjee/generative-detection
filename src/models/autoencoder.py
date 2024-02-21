import sys
latent_diffusion_path = "submodules/latent-diffusion/"
sys.path.append(latent_diffusion_path)
from ldm.models.autoencoder import AutoencoderKL

class Autoencoder(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)