# src/modules/autoencodermodules/feat_encoder.py
from ldm.modules.diffusionmodules.model import Encoder as LDMEncoder

class FeatEncoder(LDMEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        