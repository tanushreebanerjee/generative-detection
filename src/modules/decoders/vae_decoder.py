from ldm.modules.diffusionmodules.model import Decoder as LDMDecoder

class Decoder(LDMDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        