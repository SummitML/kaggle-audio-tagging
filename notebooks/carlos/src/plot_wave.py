import matplotlib.pyplot as plot
import numpy as np
import wave
import sys
import os

# Reference https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file

# TODO this wave extraction can become a decorator

"""
Extract raw audio
"""
# https://docs.python.org/3.6/library/wave.html
# a primer on the wav format - https://blogs.msdn.microsoft.com/dawate/2009/06/23/intro-to-audio-programming-part-2-demystifying-the-wav-format/
wv = wave.open(f'../../data/external/audio_train/0a0a8d4c.wav','rb')
# Reads and returns at most n frames of audio, as a bytes object.
# -1 likely signals all the frames
signal = wv.readframes(-1) # should return a 2 byte string

"""
Flatten audio to array
"""
# Interpret a buffer as a 1-dimensional array
np.seterr('raise') # Set how floating-point errors are handled.
signal = np.frombuffer(signal, 'int') #
# NOTE setting "data-type=int" here is critical
# to ensure array only contains integers. The
# default data type is float which will cause the array
# to include nan values, which will in turn cause an overflow err 


# Stereo? spf.getnchannels() == 2:
print(f'Channels: {wv.getnchannels()}')

"""
Visualize
"""
# https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
plot.figure(1) # creates a new figure with an id of 1
plot.title('Wave')
plot.plot(signal) # plots signal bytes as x,y?
plot.show() # Displays a figure
