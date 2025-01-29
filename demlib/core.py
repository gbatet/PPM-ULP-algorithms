# -*- coding: utf-8 -*-
"""

Author: Gerard Batet
Institution: Universitat Politècnica de Catalunya (UPC)
email: gerard.batet@upc.edu
created: 23/01/2024
"""


from math import sqrt, floor


from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import firwin, lfilter
from scipy.io import wavfile
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True
})

#######################################################
# READ DATA
#######################################################

def read_wav(file):
    sample_rate, samples = wavfile.read(file)
    time = np.arange(0, ((len(samples) - 0.5) / sample_rate), (1 / sample_rate))
    return sample_rate, samples, time


#######################################################
# EMULATE ADC OPERATIVE
#######################################################

def do_downsample_float(data, factor):
    """
    Downsample a signal by a specified float  factor

    Parameters:
    - data: np.array, the input signal to be downsampled
    - factor: int, the downsampling factor (every `factor`-th sample is kept)

    Returns:
    - downsampled_signal: np.array, the downsampled signal
    """
    downsampled_signal = []

    # Check for downsampling factor
    if factor <= 1:
        raise ValueError("Downsampling factor must be greater than 1.")

    # Downsample by taking every `factor`-th sample from the filtered signal
    i = 0
    while i < len(data):
        downsampled_signal.append(data[floor(i)])
        i = i + factor

    return downsampled_signal


def do_buffers(data, length):
    buffer = []
    time_buf = []

    while len(data) % length != 0:
        data.append(np.mean(data))

    for i in range(int(len(data) / length)):
        buffer.append([])
        time_buf.append(i * length)
        for j in range(length):
            buffer[i].append(data[i * length + j])

    print(len(buffer), len(time_buf))

    return buffer, time_buf


#######################################################
# PROCESS THE BUFFER
#######################################################

# TEMPORAL
###########################


# SPECTRAL
###########################

def do_fft(data, sample_rate,lb, ub):

    """
    Implements the fft in a given signal buffer sample array.

    Parameters:
    data (np.array): Input signal samples in buffers of a given length.
    sample_rate (float): Sampling rate in Hz.
    lb (int): left cut Frequency of FFT in Hz.
    ub (int): right cut Frequency of FFT in Hz.

    Returns:
    out_fft (np.array): Output array for each buffer FFT.
    """
    out_fft = []

    # Y (array): Power at a given frequency.
    # Xfft (array): Frequency bin.

    for i in range(len(data)):
        N = len(data)
        T = 1 / sample_rate

        Yfft = fft(data)
        Y = 2.0/N * np.abs(Yfft[0:N//2])

        Xfft = fftfreq(N, T)[:N//2]
        Y[0] = 0

        out_fft.append(max(Y[lb:ub]))

    return out_fft



#######################################################
# DETECT THE SIGNAL
#######################################################

# MANAGE THRESHOLD
###########################

