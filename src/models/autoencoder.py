# src/models/autoencoder.py

from ldm.models.autoencoder import AutoencoderKL

class Autoencoder(AutoencoderKL):
    """Autoencoder model with KL divergence loss."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)