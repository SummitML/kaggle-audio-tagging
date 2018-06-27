import os
import csv

def paths_for_tags(csv_path:str, filters:list=[]) -> list:
    """
    Returns a list a of file paths
    filtered by a specified list of tags

    Usage:
    saxophone_paths = paths_for_tags('path/to/file.csv', ['Saxophone'])
    # => ['1234.wav']
    """
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        files = []
        for i, row in enumerate(reader):
            # skip labels row
            if i > 0:
                fname, tag, _ = row
                files.append({ 'file': fname, 'tag': tag })

        if not len(filters):
            return files
        else:
            return [ f['file'] for f in files if f['tag'] in filters]
