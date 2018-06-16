%matplotlib inline
import matplotlib.pyplot as plot
import numpy as np
import wave
import sys
import os
from datetime import datetime

# Reference https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file

def plot_wave(path:str):
    """
    Extract raw audio, flatten audio to array, then plot
    """
    # https://docs.python.org/3.6/library/wave.html
    # a primer on the wav format - https://blogs.msdn.microsoft.com/dawate/2009/06/23/intro-to-audio-programming-part-2-demystifying-the-wav-format/
    wv = wave.open(path,'rb')

    # Reads and returns at most n frames of audio, as a bytes object.
    signal = wv.readframes(-1)
    # -1 likely signals all the frames
    # should return a 2 byte string

    # Set how floating-point errors are handled.
    np.seterr('raise')

    # Interpret a buffer as a 1-dimensional array
    signal = np.frombuffer(signal, 'int') #

    # NOTE setting "data-type=int" here is critical
    # to ensure array only contains integers. The
    # default data type is float which will cause the array
    # to include nan values, which will in turn cause an overflow err


    # Stereo? wv.getnchannels() == 2:
    print(f'Channels: {wv.getnchannels()}')

    # https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
    plot.figure(1) # creates a new figure with an id of 1
    plot.title('Wave')
    plot.plot(signal) # plots signal bytes as x,y?
    plot.show() # Displays a figure



def plot_waves(folder_path):
    """
    Plot waves from directory
    """
    started_at = datetime.now()
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            try:
                plot_wave(path)
            except Exception as e:
                print(f'{path} failed. {e}')

    completed = started_at - datetime.now()
    print(f'completed in {completed.seconds}')

# plot_wave('../../data/external/audio_train/0a0a8d4c.wav')
plot_waves('../../data/external/audio_train')
