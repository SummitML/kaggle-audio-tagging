import sys
import os
from datetime import datetime
import wave
import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Reference https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file

def plot_wave(path:str, as_spectogram=False):
    """
    Extract raw audio, flatten audio to array, then plot
    """
    # https://docs.python.org/3.6/library/wave.html
    # a primer on the wav format - https://blogs.msdn.microsoft.com/dawate/2009/06/23/intro-to-audio-programming-part-2-demystifying-the-wav-format/
    wv = wave.open(path,'rb')
    # Stereo? wv.getnchannels() == 2:
    print(f'Channels: {wv.getnchannels()}')

    # Reads and returns at most n frames of audio, as a bytes object.
    frames = wv.readframes(-1)
    # -1 likely framess all the frames
    # should return a 2 byte string

    # Set how floating-point errors are handled.
    np.seterr('raise')

    # Interpret a buffer as a 1-dimensional array
    sound_bytes = np.frombuffer(frames, 'int') #
    # NOTE setting "data-type=int" here is critical
    # to ensure array only contains integers. The
    # default data type is float which will cause the array
    # to include nan values, which will in turn cause an overflow err

    # spectrogram
    if as_spectogram:
        f, t, Sxx = signal.spectrogram(sound_bytes, 10e3)
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        # return

    # https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
    plt.figure(1) # creates a new figure with an id of 1
    plt.title('Wave')
    plt.plot(sound_bytes) # plots frames bytes as x,y?
    plt.show() # Displays a figure


def plot_waves(folder_path='', as_spectogram=False):
    """
    Plot waves from directory
    """
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            try:
                plot_wave(path, as_spectogram)
            except Exception as e:
                print(f'{path} failed. {e}')

# plot_wave('../../data/external/audio_train/0a0a8d4c.wav')
plot_waves('../../data/external/sample', as_spectogram=True)
print(timeit.timeit(plot_waves))



#TODO convert extract sine wave from wav
