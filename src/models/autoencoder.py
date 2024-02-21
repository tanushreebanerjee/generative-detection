# src/models/autoencoder.py

from ldm.models.autoencoder import AutoencoderKL

class Autoencoder(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)