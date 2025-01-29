# -*- coding: utf-8 -*-
"""

Author: Gerard Batet
Institution: Universitat Politècnica de Catalunya (UPC)
email: gerard.batet@upc.edu
created: 23/01/2024
"""

# IMPORTS

import demlib as dm
from docs.filter_coefficients import coefficients_antialiassing, coefficients_bandpass

from argparse import ArgumentParser
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True
})

# MAIN

def main(args):
    #######################################################
    # VARIABLES
    #######################################################
    method = ""  # Variable for method of processing
    method_units = ""  # Variable for method units
    method_res = []  # Variable for the  method application result
    method_times = []  # Variable to store the times vector

    #######################################################
    # READ DATA
    #######################################################
    sample_rate, samples, time = dm.read_wav(args.input)  # Samples in counts

    #######################################################
    # EMULATE ADC OPERATIVE
    #######################################################

    # DownSample
    factor = sample_rate / args.sampleRate
    samples_down, times_down = dm.do_downsample_float(samples, args.sampleRate, factor)

    # Create buffers
    buffer, buf_times = dm.do_buffers(samples_down, args.buffer, args.sampleRate)


    #######################################################
    # PROCESS THE BUFFERS
    #######################################################

    # FFT
    if args.method == 1:

        method_units = r"\bf{Spectral Power}"
        method_res, lb, ub = dm.do_fft(buffer, args.sampleRate, args.onFreq, args.offFreq)
        method_times = buf_times

        method = r"\bf{FFT}"
        print("Method: FFT")
        print(f"Set frequency bandpass: {args.onFreq} -{args.offFreq} Hz ")
        print(f"Real frequency bandpass: {lb} -{ub} Hz ")

    # Goertzel
    elif args.method == 2:
        method_units = r"\bf{Spectral Power}"
        method_res = dm.do_goertzel(buffer, args.sampleRate, args.goertzel)
        method_times = buf_times

        method = r"\bf{GOERTZEL}"
        print("Method: GOERTZEL")
        print(f"Center set frequency: {args.goertzel} Hz ")
        print(f"Frequency bin width: {args.sampleRate/(args.buffer*2)} Hz ")

    # Filtering
    elif args.method == 3:
        method_units = r"\bf{Counts}"
        method_res, method_times = dm.filter(buffer,args.sampleRate, args.filterFrequency, args.localOscillator, coefficients_antialiassing, coefficients_bandpass)

        method = r"\bf{FILTERING}"
        print("Method: FILTERING")

    # Invalid method
    else:
        print("Invalid method")

    # CFAR THRESHOLD
    # TOA DETECTION

    # PLOT
    # # ADC input

    fig_gnrl, (gnrl) = plt.subplots()
    gnrl.plot(time, samples)
    gnrl.set_ylabel(r"\bf{ADC Counts - x(n)}")
    gnrl.set_xlabel(r"\bf{Time (s)}")

    # # Detection method
    fig_method, (mth) = plt.subplots()
    mth.plot(method_times, method_res)
    mth.set_ylabel(method + " - " + method_units)
    mth.set_xlabel(r"\bf{Time (s)}")

    # # PLT SHOW
    plt.show()

    return None

if __name__ == "__main__":

    # INIT
    argparser = ArgumentParser()

    # INFO
    argparser.add_argument("-v", "--verbose", help="more info", action="store_true")

    # INPUT
    argparser.add_argument("-i", "--input", help="Input file directory and name, default docs/1.wav", type=str, default="docs/1.wav")

    # ACQUISITION
    argparser.add_argument("-sr", "--sampleRate", help="Sample Rate of the uC, default 150000 S/s", type=int, default=150000)
    argparser.add_argument("-b", "--buffer", help="ADC Buffer lenght, default = 256", type=int, default=256)

    # PROCESSING
    argparser.add_argument("-m", "--method", help="Processing method, 1: FFT, 2: Go, 3: Filt, default = 1", type = int, default=1)

    # CONFIG METHODS
    argparser.add_argument("-goe", "--goertzel", help="Goertzel method, center frequency, default = 69000", type=int, default=69000)

    argparser.add_argument("-on", "--onFreq", help="Bandpass on frequency in Hz, default 68000 Hz", type=int, default = 68000)
    argparser.add_argument("-off", "--offFreq", help="Bandpass off frequency in Hz, default 70000 Hz", type=int, default=70000)

    argparser.add_argument("-ffreq", "--filterFrequency", help="center signal frequency", type=int, default=69000)
    argparser.add_argument("-LO", "--localOscillator", help="Local Oscillator, default 58000", type=int, default=58000)

    # OUTPUT
    argparser.add_argument("-sw", "--show", help="show plot default YES (1)", type=int,default=1)
    argparser.add_argument("-o", "--output", help="output file, default out.txt", type=str, default="out.txt")

    # EXIT
    args = argparser.parse_args()


    try:
        main(args)
    except KeyboardInterrupt:
        print("Program terminated by user.")
