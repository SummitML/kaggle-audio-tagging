import pickle
import os

def pickle_out(filename:str, data:dict):
    """

    """
    try:
        # create a directory if it doesn't exist
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, 'wb') as outfile:
            pickle.dump(data, outfile)
    except Exception as e:
        print(f'Something went wrong... {e}')
        raise e
