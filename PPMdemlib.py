# -*- coding: utf-8 -*-
"""

Author: Gerard Batet
Institution: Universitat Politècnica de Catalunya (UPC)
email: gerard.batet@upc.edu
created: 31/10/2024
"""

import numpy as np

from math import sqrt, floor


from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import firwin, lfilter



##########################################################
# PREP FUNC
##########################################################

# NORMALIZATION


def normalize_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def normalize_neg1_1(data):
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1


def normalize_custom(data, a, b):
    return a + (data - np.min(data)) * (b - a) / (np.max(data) - np.min(data))

# DOWNSAMPLING


def do_downsample_integer(data, factor):
    """
    Downsample a signal by a specified integer factor

    Parameters:
    - data: np.array, the input signal to be downsampled
    - factor: int, the downsampling factor (every `factor`-th sample is kept)

    Returns:
    - downsampled_signal: np.array, the downsampled signal
    """
    # Check for downsampling factor
    if factor <= 1:
        raise ValueError("Downsampling factor must be greater than 1.")

    # Downsample by taking every `factor`-th sample from the filtered signal
    downsampled_signal = data[::factor]

    return downsampled_signal

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


##########################################################
# FREQUENCY BASED
##########################################################

# HAHN WINDOWING


def init_hann(data_len):

    return signal.windows.hann(data_len)


def do_hann(data, win):

    for i in range(len(data)):
        data[i] = data[i] * win[i]
    return data

# FFT


def do_fft(data, sample_rate):

    """
    Implements the fft in a given signal sample array.

    Parameters:
    data (np.array): Input signal samples.
    sample_rate (float): Sampling rate in Hz.

    Returns:
    Y (array): Power at a given frequency.
    Xfft (array): Frequency bin.
    """

    N = len(data)
    T = 1 / sample_rate

    Yfft = fft(data)
    Y = 2.0/N * np.abs(Yfft[0:N//2])

    Xfft = fftfreq(N, T)[:N//2]

    return Y, Xfft

# GÖERTZEL


def do_goertzel(data, sample_rate, target_freq):

    """
    Implements the Goertzel algorithm to detect a specific target frequency
    in a given signal sample array.

    Parameters:
    samples (np.array): Input signal samples.
    sample_rate (float): Sampling rate in Hz.
    target_freq (float): Target frequency to detect in Hz.

    Returns:
    float: Power at the target frequency.
    """
    # Calculate normalized frequency and Goertzel coefficient
    omega = 2 * np.pi * target_freq / sample_rate
    coeff = 2 * np.cos(omega)

    # Initialize Goertzel variables
    s_prev = 0.0
    s_prev2 = 0.0

    # Run Goertzel on the sample set
    for sample in data:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    # Calculate the power at the target frequency
    power = s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2
    return power

# WAVELET


def init_wavelet(sample_rate, freq, periodes):

    step = 2*np.pi/(sample_rate/freq)

    wavelet_pre1 = -np.sin(np.arange(0, periodes * 2 * np.pi, step))
    wavelet_pre2 = np.cos(np.arange(0, periodes * 2 * np.pi, step))

    window = init_hann(len(wavelet_pre1))

    wavelet1 = wavelet_pre1 * window
    wavelet2 = wavelet_pre2 * window

    return wavelet1, wavelet2

def do_wavelet(data, wavelet1,wavelet2):

    dataSin = np.convolve(data, wavelet1)

    dataCos = np.convolve(data, wavelet2)
    vector = []

    for i in range(len(dataSin)):
        vector.append(sqrt((dataCos[i]*dataCos[i]+dataSin[i]*dataSin[i])))

    result = max(vector)
    return result
##########################################################
# TEMPORAL BASED
##########################################################


# DOWNCONVERSION


def mix_downconvert(data, sample_rate, f_LO):
    """
    Downconvert a signal by mixing with a local oscillator and low-pass filtering.

    Parameters:
    - signal: np.array, the input signal to be downconverted
    - fs: float, the sampling frequency of the signal
    - f_LO: float, the frequency of the local oscillator for downconversion

    Returns:
    - downconverted_signal: np.array, the downconverted signal
    """

    # Generate the local oscillator (cosine wave at f_LO)
    t = np.arange(len(data)) / sample_rate
    LO_signal = np.cos(2 * np.pi * f_LO * t)

    # Mix the input signal with the LO signal
    mixed_signal = data * LO_signal


    return mixed_signal


def filter_downconversion(data, sample_rate, cutoff_freq, num_taps=101):

    """
    Filter mixed signal with LO

    Parameters:
    - data: np.array, the input signal mixed with LO to be filtered
    - sample_rate: float, the sampling frequency of the signal
    - f_LO: float, the frequency of the local oscillator for downconversion
    - cutoff_freq: float, the cutoff frequency for the low-pass filter after mixing
    - num_taps: int, number of taps for the low-pass FIR filter (default: 101)

    Returns:
    - downconverted_signal: np.array, the downconverted filtered signal
    """
    # Design a low-pass FIR filter to isolate the downconverted signal (baseband)
    lowpass_filter = firwin(num_taps, cutoff=cutoff_freq, fs=sample_rate)

    # Apply the low-pass filter to the mixed signal
    downconverted_signal = lfilter(lowpass_filter, 1.0, data)

    return downconverted_signal


# GENERIC FIR FILTER


def fir_filter(signal, coefficients):
    """
    Apply an FIR filter to a signal using manually specified coefficients.

    Parameters:
    - signal: np.array, the input signal to filter
    - coefficients: np.array or list, the filter coefficients

    Returns:
    - filtered_signal: np.array, the filtered output signal
    """
    # Convolve the signal with the filter coefficients to apply the FIR filter
    filtered_signal = np.convolve(signal, coefficients, mode='same')

    return filtered_signal