from .wavs import Wav

class NormalizeAudio:
    def __init__(self, wav_inputs:list=[]):
        self.inputs = wav_inputs

    def sample_rate(self, sample_rate:int=22050) -> list:
        return list(map(lambda sample: Wav(name=sample.name, wav=sample.wav[0][:sample_rate], label=sample.label), self.inputs))
