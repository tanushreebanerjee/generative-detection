# src/modules/losses/contperceptual.py

from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator as LPIPSWithDiscriminator_LDM

class LPIPSWithDiscriminator(LPIPSWithDiscriminator_LDM):
    """LPIPS loss with discriminator."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)