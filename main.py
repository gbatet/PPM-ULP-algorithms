# -*- coding: utf-8 -*-
"""

Author: Gerard Batet
Institution: Universitat Politècnica de Catalunya (UPC)
email: gerard.batet@upc.edu
created: 08/05/2024
"""

# IMPORTS
from argparse import ArgumentParser

import pandas as pd
import pyarrow

import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, floor, ceil

import time

from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.io import wavfile

from PPMdemlib import normalize_custom, init_hann, do_hann,do_fft, do_goertzel, init_wavelet, do_wavelet, mix_downconvert, filter_downconversion, do_downsample_integer, do_downsample_float

plt.rcParams.update({
    "text.usetex": True
})

# MAIN


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("-v", "--verbose", help="more info", action="store_true")

    argparser.add_argument("-i", "--input", help="Input file directory and name, default csv/analog.csv", type=str, default="csv/analog.csv")

    argparser.add_argument("-sr", "--sampleRate", help="Sample Rate of the uC, default 150000 S/s", type=int, default=150000)
    argparser.add_argument("-b", "--buffer", help="ADC Buffer lenght, default = 128", type=int, default=128)

    argparser.add_argument("-on", "--onFreq", help="Bandpass on frequency in Hz, default 68000 Hz", type=int, default = 68000)
    argparser.add_argument("-off", "--offFreq", help="Bandpass off frequency in Hz, default 70000 Hz", type=int, default=70000)

    argparser.add_argument("-sw", "--show", help="show plot default no (0)", type=int,default=0)
    argparser.add_argument("-o", "--output", help="output file, default out.txt", type=str, default="out.txt")
    args = argparser.parse_args()

# FILES & VARIABLES

    # CSV READ
    # df = pd.read_csv(args.input)
    #
    # time = df["Time [s]"]
    # samples = df["TP2"]
    #
    # sample_rate = 1/(time[1]-time[0])

    #WAV READ

    sample_rate, samples = wavfile.read(args.input)

    time = [range(0,sample_rate,int(len(samples)/sample_rate))]

    # PREPARING THE BUFFER
    #########################################################################################
    # Matching voltage levels

    samples = normalize_custom(samples, 1, -1) + 1.2

    # Matching the sample rate of the uC if the file has a bigger sample rate than args.sampleRate

    sample_uC = []
    sampleDiv = sample_rate / args.sampleRate

    sample_uC = do_downsample_float(samples, sampleDiv)

    # Matching the buffer operation of the uC

        # Making the buffer len divisable by the buffer len
    while len(sample_uC) % args.buffer != 0:
        sample_uC.append(1.2)

        # Extracting time steps to later plot
    xSample = []
    step = 1/args.sampleRate

    for i in range(len(sample_uC)):
        xSample.append(i*step)

        #Extracting the timestamps of the buffer init for plot
    xBuffer = []
    stepBuffer = args.buffer*step
    for i in range(int(len(sample_uC)/args.buffer)-1):
        xBuffer.append(i * stepBuffer)


    #########################################################################################

# PROCESSING THE BUFFER

    # START general DFFT config

    fft_max = []

    lb = floor(args.onFreq * args.buffer / args.sampleRate)
    ub = ceil(args.offFreq * args.buffer / args.sampleRate)

    # END general DFFT config

    # START Hann window DFFT config

    window = init_hann(args.buffer)

    adc_hann = []
    fft_hann_max = []

    # END Hann window DFFT config

    # START Wavelet config

    wavelet1, wavelet2 = init_wavelet(args.sampleRate, 69000, 3.5)
    wave_data = []
    lenWave = int(len(wavelet1) / 2)

    # END Wavelet config

    # START Goertzel config

    goertzel_data = []

    # END Goertzel config

    for n in range(int(len(sample_uC)/args.buffer)-1):

        #######################################################

        adc_buf = sample_uC[args.buffer*n:args.buffer*(n+1)]

        #######################################################

        # START FFT BASIC ALGORITHM

        fft_res, xf = do_fft(adc_buf, args.sampleRate)
        fft_res[0] = 0

        fft_max.append(max(fft_res[lb:ub]))

        # END DFFT BASIC ALGORITHM

        # START DFFT WITH HANN WINDOW ALGORITHM

        adc_hann = do_hann(adc_buf, window)

        fft_hann_res, xf = do_fft(adc_hann, args.sampleRate)

        fft_hann_res[0] = 0
        fft_hann_max.append(max(fft_hann_res[lb:ub]))

        # END DFFT WITH HANN WINDOW ALGORITHM

        # START GOERTZEL ALGORITHM

        data_prov = do_goertzel(adc_buf, args.sampleRate, 69000)

        goertzel_data.append(data_prov)

        # END GOERTZEL

        #START WAVELET

        wavelet = do_wavelet(adc_buf,wavelet1,wavelet2)

        wave_data.append(wavelet)

        # END_WAVELET

        #


# PLOT

    # FIG configuration
    fig, (AIN, FFT, GO, WAVE) = plt.subplots(4, 1)

    fig.suptitle(r"\bf{PPM reception algorithm comparison}")
    fig.set_figheight(9.5)
    fig.set_figwidth(7.5)
    fig.subplots_adjust(hspace=0.47)

    # ADC input samples plot
    AIN.plot(xSample, sample_uC)

    AIN.set_title(r"Sample data fed into $\mu C$")
    AIN.set_xlabel(r"\bf{Seconds (s)}")
    AIN.set_ylabel(r"\bf{Volts (V)}")
    AIN.set_ylim(0, 3.3)

    # Simple DFFT plot
    FFT.plot(xBuffer, fft_max)

    FFT.set_title(rf"DFFT {args.onFreq/1000} kHz - {args.offFreq/1000} kHz band max value")
    FFT.set_xlabel(r"\bf{Buffer start time (s)}")
    FFT.set_ylabel(r"\bf{Power spectral density}")
    FFT.set_ylim(0, max(fft_max)*1.05)

    # GOERTZEL PLOT
    GO.plot(xBuffer, goertzel_data)
    GO.set_title(r"Göertzel algorithm for 69 kHz")
    GO.set_xlabel(r"\bf{Seconds (s)}")
    GO.set_ylabel(r"\bf{Power spectral density}")


    # wavelet plot

    WAVE.plot(wave_data)
    WAVE.set_title(r"Gaussian wavelet convolution at 69 kHz")
    WAVE.set_xlabel(r"\bf{Seconds (s)}")
    WAVE.set_ylabel(r"\bf{Power spectral density}")


    # PLT SHOW
    plt.show()

