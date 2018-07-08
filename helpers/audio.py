from .wavs import Wav
import numpy as np
from functools import partial

class NormalizeAudio:
    _isWavInstance = False

    def __init__(self, wav_inputs:list=[]):
        # can accept Wav instance or dict, normalize to dict
        if isinstance(wav_inputs[0], Wav):
            self.inputs = list(map(lambda x: x.__dict__, wav_inputs))
            self._isWavInstance = True
        else:
            self.inputs = wav_inputs

    # TODO
    def _pad_sound(self, wav:np.ndarray, sample_rate:int):
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
        padded_sound = np.tile(wav, math.ceil(sample_rate / wav.shape[0]))
        return padded_sound[:sample_rate]

    def _normalize_sample_rate(self, sample_rate:int, wav:np.ndarray,):
        wav_ndarr, _ = wav.get('wav')
        output = Wav(name=wav.get('name'),
                     wav=(wav_ndarr[:sample_rate], _),
                     label=wav.get('label'))

        # output same as input
        if self._isWavInstance:
            return output
        else:
            return output.__dict__

    def sample_rate(self, sample_rate:int=22050) -> list:
        fn = partial(self._normalize_sample_rate, sample_rate)
        return list(map(fn, self.inputs))
