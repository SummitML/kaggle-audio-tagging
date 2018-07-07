import os
import pickle
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy import signal, array
import librosa

class Wav:
    def __init__(self, name:str='', wav:np.ndarray=None, label:str=''):
        self.name = name
        self.wav = wav
        self.label = label

def _load_wav(duration:int, wav:dict) -> Wav:
    """
    Generates a namedtuple with some metadata
    and a librosa loaded wav file (array)
    """
    print(f'{wav["name"]}...')
    librosa_loaded_wav = librosa.load(wav['path'], duration=duration)
    return Wav(name=wav['name'], wav=librosa_loaded_wav, label=wav['tag'])

def load_wav_files(paths:list, duration:int=15) -> list:
    """
    Generates list of dictionaries
    with loaded wave files (leveraging librosa)

    Usage:
    train_wav_inputs = load_wav_files(paths)
    # => [
        Wav(name='00353774.wav', wav=(array()),
    ]
    """
    fn = partial(_load_wav, duration)
    with ProcessPoolExecutor() as executor:
        return list(executor.map(fn, paths))

def _deserialize(item:Wav) ->  Wav:
    return pickle.loads(item)

def deserialize_wavs(wavs:list) -> list:
    """

    """
    with ProcessPoolExecutor() as executor:
        return list(executor.map(_deserialize, wavs))

def _serialize(item:Wav) -> Wav:
    return pickle.dumps(item, protocol=0)

def serialize_wavs(wavs:list) -> list:
    """

    """
    with ProcessPoolExecutor() as executor:
        return list(executor.map(_serialize, wavs))

def _load_pickled_wavs(folder_path:str, path:str):
    """

    """
    with open(f'{folder_path}/{path}', 'rb') as f:
        print(f'Loading {path}...¯\_(ツ)_/¯')
        return pickle.loads(f.read())

def assemble_batched_wavs(folder_path:str) -> list:
    """

    """
    fn = partial(_load_pickled_wavs, folder_path)
    for root, dirs, files in os.walk(folder_path):
        with ProcessPoolExecutor() as executor:
            return list(executor.map(fn, files))

    return []
