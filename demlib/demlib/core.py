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



class Demodulate:

    #######################################################
    # READ DATA
    #######################################################
    @staticmethod
    def read_wav(filename):
        sample_rate, samples = wavfile.read(filename)
        time = []
        return sample_rate, samples, time

    #######################################################
    # EMULATE ADC OPERATIVE
    #######################################################
    @staticmethod
    def do_downsample_float(data, factor):
        return data

    @staticmethod
    def do_buffers(data, buf_size):
        return data



