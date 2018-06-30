import os
import csv
import librosa
from concurrent.futures import ProcessPoolExecutor

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


def _load_wav(wav:dict) -> dict:
    print(f'{wav["name"]}...')
    return { wav['name']: librosa.load(wav['path']) }

def load_wav_files(paths:list) -> list:
    """
    Returns a list of dictionaries
    with loaded wave files

    Usage:
    train_wav_inputs = load_wav_files(paths)
    # => [{ '1234.wav': array([]) }]
    """

    with ProcessPoolExecutor() as executor:
        return list(executor.map(_load_wav, paths))
