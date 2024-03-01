from ldm.modules.diffusionmodules.model import Encoder as LDMEncoder

class Encoder(LDMEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)