# src/modules/autoencodermodules/feat_decoder.py
from ldm.modules.diffusionmodules.model import Decoder as LDMDecoder

class FeatDecoder(LDMDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
