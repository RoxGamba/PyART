import numpy as np; import os
from ..waveform import  Waveform


class Waveform_GRA(Waveform):
    """
    Class to handle GRAthena++ waveforms.
    This is still under development, depending on the
    final format of the data.
    For now, it assumes that the data is in the format
    of Alireza's simulations
    """

    def __init__(
            self,
            path   ='../dat/GRA/',
        ):

        super().__init__()
        self.path = path
        pass

    def load_metadata(self):
        pass

    def load_hlm(self):
        pass
    