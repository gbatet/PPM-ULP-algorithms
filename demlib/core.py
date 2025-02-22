# -*- coding: utf-8 -*-
"""

Author: Gerard Batet
Institution: Universitat Politècnica de Catalunya (UPC)
email: gerard.batet@upc.edu
created: 23/01/2024
"""


from math import floor, ceil, trunc


from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann, blackman, blackmanharris
from scipy.io import wavfile
import numpy as np


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

def do_downsample_float(data, sample_rate, factor):
    """
    Downsample a signal by a specified float  factor

    Parameters:
    - data: np.array, the input signal to be downsampled
    - sample_rate: int, the final sampling rate
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

    times = np.arange(0, len(downsampled_signal))
    times = times/sample_rate

    return downsampled_signal, times


def do_buffers(data, length, sample_rate):
    data_buf = data[:]
    buffer = []
    time_buf = []

    while len(data_buf) % length != 0:
        data_buf.append(np.mean(data_buf))

    for i in range(int(len(data_buf) / length)):
        buffer.append([])
        time_buf.append(i * length / sample_rate)
        for j in range(length):
            buffer[i].append(data_buf[i * length + j])

    return buffer, time_buf


#######################################################
# PROCESS THE BUFFER
#######################################################

# TEMPORAL
###########################
# Down convert signal mixed with set LO
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

def filter_antialiassing(data, coefficients):

    """
    Filter mixed signal with LO

    Parameters:
    - data: np.array, the input signal mixed with LO to be filtered
    - sample_rate: float, the sampling frequency of the signal
    - f_LO: float, the frequency of the local oscillator for downconversion
    - cutoff_freq: float, the cutoff frequency for the low-pass filter after mixing
    - num_taps: int, number of taps for the low-pass FIR filter (default: 5)

    Returns:
    - downconverted_signal: np.array, the downconverted filtered signal
    """
    filtered_signal = np.convolve(data, coefficients, mode='same')

    return filtered_signal


def do_filter(data, sample_rate, frequency, f_lo, coefficients_antialiassing, coefficients_bandpass):

    factor = trunc(frequency / (frequency - f_lo))

    filter_output = []

    for i in range(len(data)):
        mixed_signal = mix_downconvert(data[i], sample_rate, f_lo)
        prefiltered_signal = filter_antialiassing(mixed_signal, coefficients_antialiassing)
        new_signal = prefiltered_signal[::factor]
        filtered_signal = np.convolve(new_signal, coefficients_bandpass, mode='same')

        filter_output.append(np.mean(abs(filtered_signal)))


    return filter_output

# SPECTRAL
###########################

# FFT


def do_fft(data, sample_rate,onFreq, offFreq):

    """
    Implements the fft in a given signal buffer sample array.

    Parameters:
    data (np.array): Input signal samples in buffers of a given length.
    sample_rate (float): Sampling rate in Hz.
    onFreq (int): left cut Frequency of FFT in Hz.
    offFreq (int): right cut Frequency of FFT in Hz.

    Returns:
    out_fft (np.array): Output array for each buffer FFT.
    """
    out_fft = []

    # Y (array): Power at a given frequency.
    # Xfft (array): Frequency bin.

    lb = floor(onFreq * len(data[0]) / sample_rate)
    ub = ceil(offFreq * len(data[0]) / sample_rate)

    N = len(data[0])
    T = 1 / sample_rate
    w = blackman(N)

    for i in range(len(data)):
        Yfft = fft(data[i]*w)
        Y = 2.0/N * np.abs(Yfft[0:N//2])
        Y[0] = 0
        out_fft.append(np.mean(Y[lb:ub]))

    Xfft = fftfreq(N, T)[:N // 2]
    rlb = Xfft[lb]
    rub = Xfft[ub]

    return out_fft, rlb, rub

# Goertzel


def do_goertzel(data, sample_rate, target_freq):

    """
    Implements the Goertzel algorithm to detect a specific target frequency
    in a given signal sample array.

    Parameters:
    data (np.array): Input signal samples.
    sample_rate (float): Sampling rate in Hz.
    target_freq (float): Target frequency to detect in Hz.

    Returns:
    float: Power at the target frequency.
    """
    # Calculate normalized frequency and Goertzel coefficient
    omega = 2 * np.pi * target_freq / sample_rate
    coeff = 2 * np.cos(omega)

    power = []
    # Run Goertzel
    for i in range(len(data)):
        # Initialize Goertzel variables
        s_prev = 0.0
        s_prev2 = 0.0
        for sample in data[i]:
            s = sample + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s

        # Calculate the power at the target frequency
        power.append(s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2)

    return power

#######################################################
# DETECT THE SIGNAL
#######################################################

# MANAGE THRESHOLD
###########################


def do_cfar(data, sample_rate, buffer_len, pulse_width, cells):

    """
    Implements CFAR to detect a signal

    Parameters:
    data (list): Input signal samples.
    sample_rate (float): Sampling rate in Hz.
    buffer_len (float)
    pulse_width (int): width time of the pulse to calculate the guard time
    cells (int): number of cells to calculate CFAR

    Returns:
    threshold (list): Calculated threshold for each index of the list
    detec (list): List with 0-1 if the data is under the threshold
    """

    threshold = []
    detect = []
    guard = 5*ceil(pulse_width * sample_rate / (buffer_len * 1000))

    for i in range(len(data)):
        if data[i] == 0:
            res = 1
        elif i < (cells + guard + 1):
            res = np.mean(data[(i+guard+1):(i+guard+cells)])/data[i]

        elif i > (len(data)-cells-guard-1):
            res = np.mean(data[(i-guard-cells):(i-guard-1)])/data[i]

        else:
            do_mean= data[(i-guard-cells):(i-guard-1)] + data[(i+guard+1):(i+guard+cells)]
            res = np.mean(do_mean)/data[i]

        if data[i] > res:
            detect.append(1)
        else:
            detect.append(0)

        threshold.append(res)

    return threshold, detect

# CHECK PULSE WIDTH
############################


def do_check_pulse(data, sample_rate, buffer_length, pulse_width):

    """
    Checks if a pulse is correct within pm 2ms

    Parameters:
    data (list): Input signal detection samples.
    sample_rate (float): Sampling rate in Hz.
    buffer_length (float)
    pulse_width (int): width time of the pulse

    Returns:
    detec (list): List with 0-1 if the pulse is correct
    detect_times (list): Times wehere the pulse ends
    """

    detect = []
    time_buf = buffer_length/sample_rate
    up_ant = 0
    up = 0
    detect_times = [-1]
    pulse_width = pulse_width/1000

    for i in range(len(data)):

        if data[i] == 1 and data [i-1] == 0:

            up_ant = up
            up = i
            detect.append(-1)

        elif data[i] == 0 and data [i-1] == 1:

            prob = (i - up)*time_buf
            prob_ant = (i - up_ant)*time_buf

            if pulse_width+0.002 >= prob >= pulse_width-0.002:
                detect.append(1)
                detect_times.append(i*time_buf)

            elif pulse_width+0.002 >= prob_ant >= pulse_width-0.002 and (i*time_buf-detect_times[-1] > 0.1):
                detect.append(1)
                detect_times.append(i * time_buf)

            else:
                detect.append(-1)
        else:
            detect.append(-1)

    detect_times.pop(0)

    return detect, detect_times

# DECODE AND DETECTION
############################


def decode_times(data, init_time, dict_msg):

    """
    Time between pulse decoding in front of a dictionary

    Parameters:
    data (list): list of timestamps where the end of the pings are
    init_time (float in seconds): Typically 0.340 when vemco
    dict_msg (dictionary): change from decoded times to msg decoding

    Returns:
    Pings (int): Number of existing pings
    msg (list): decoded msg
    """

    init = round(init_time, 2)*100
    msg = []
    pings = len(data)

    if len(data) > 2:

        for i in range(1,len(data)):

            diff = data[i]-data[i-1]
            diff = np.round(diff, 2)*100
            # If init time pm 5ms
            if init == diff:
                msg.append("init")

            # If else that is in the dictionary
            elif (diff % 2 == 0) and (init < diff <= max(dict_msg)):
                msg.append(dict_msg[diff])

            elif 2*max(dict_msg) > diff > 2*min(dict_msg):
                msg.append("null")
                msg.append("null")
            else: pass

    else:
        msg = "Not enough data"

    return pings, msg
