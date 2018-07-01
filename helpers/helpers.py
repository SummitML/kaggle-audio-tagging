import os
import csv
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import namedtuple

import librosa
from scipy import signal, array
import numpy as np

def find_paths_with_tags(csv_path:str, files_path:str, filters:list=[], limit:int=None) -> list:
    """
    Returns a list a of file paths
    filtered by a specified list of tags

    Usage:
    from src.util import find_paths_with_tags
    saxophone_paths = find_paths_with_tags(csv_path='path/to/file.csv',
                                           files_path='path/to/files',
                                           filters=['Saxophone'])
    # => ['1234.wav']
    """
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        files = []
        for i, row in enumerate(reader):
            # skip labels row
            if i > 0:
                fname, tag, _ = row
                path_name = f'{files_path}/{fname}'
                valid_path = os.path.isfile(path_name)
                if valid_path:
                    if not filters or filters and tag in filters:
                        files.append({ 'path': path_name, 'name': fname, 'tag': tag })

            if limit is not None and len(files) > 0 and limit == len(files):
                break

        return files


Wav = namedtuple('Wav', 'name wav')

def _load_wav(duration:int, wav:dict) -> namedtuple:
    """
    Generates a namedtuple with some metadata
    and a librosa loaded wav file (array)
    """
    print(f'{wav["name"]}...')

    # foo = librosa.load('../../data/external/audio_train/01302128.wav')
    # print(foo[0])
    # print(foo[0].shape)

    librosa_loaded_wav = librosa.load(wav['path'], duration=duration)
    return Wav(name=wav['name'], wav=librosa_loaded_wav)

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
