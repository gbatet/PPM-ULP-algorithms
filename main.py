# -*- coding: utf-8 -*-
"""

Author: Gerard Batet
Institution: Universitat Politècnica de Catalunya (UPC)
email: gerard.batet@upc.edu
created: 23/01/2024
"""

# IMPORTS
import sys
import demlib as dm
from complementary.PPMdemlib_old import normalize_0_1
from demlib import do_cfar
from docs.filter_coefficients import coefficients_antialiassing, coefficients_bandpass

from math import ceil

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
    if args.output:
        sys.stdout = open("docs/log.txt", "w")

    method = ""  # Variable for method of processing
    method_res = []  # Variable for the  method application result
    method_times = []  # Variable to store the times vector

    #######################################################
    # READ DATA
    #######################################################
    sample_rate, samples, time = dm.read_wav(args.input)  # Samples in counts
    print(f"Input file: {args.input}")

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
        method_res, lb, ub = dm.do_fft(buffer, args.sampleRate, args.onFreq, args.offFreq)
        method_res_norm = normalize_0_1(method_res)
        method_times = buf_times

        method = r"\bf{FFT}"
        print("Method: FFT")
        print(f"Set frequency bandpass: {args.onFreq} -{args.offFreq} Hz ")
        print(f"Real frequency bandpass: {lb}-{ub} Hz ")

    # Goertzel
    elif args.method == 2:
        method_res = dm.do_goertzel(buffer, args.sampleRate, args.goertzel)
        method_res_norm = normalize_0_1(method_res)
        method_times = buf_times

        method = r"\bf{GOERTZEL}"
        print("Method: GOERTZEL")
        print(f"Center set frequency: {args.goertzel} Hz ")
        print(f"Frequency bin width: {args.sampleRate/(args.buffer*2)} Hz ")

    # Filtering
    elif args.method == 3:
        method_res= dm.filter(buffer,args.sampleRate, args.filterFrequency, args.localOscillator, coefficients_antialiassing, coefficients_bandpass)
        method_res_norm = normalize_0_1(method_res)
        method_times = buf_times

        method = r"\bf{FILTERING}"
        print("Method: FILTERING")

    # Invalid method
    else:
        print("Invalid method")

    #######################################################
    # THRESHOLD
    #######################################################

    threshold, detect = do_cfar(method_res_norm, args.sampleRate, args.buffer, args.pulseWidth, 50)

    #######################################################
    # DETECTION
    #######################################################

    detec_pulse, detect_times = dm.do_check_pulse(detect,args.sampleRate,args.buffer,args.pulseWidth)
    print(detect_times)
    #######################################################
    # PLOT
    #######################################################
    if args.show:
        # ADC input
        fig_gnrl, (gnrl) = plt.subplots()
        gnrl.plot(time, samples)
        gnrl.set_ylabel(r"\bf{ADC Counts - x(n)}")
        gnrl.set_xlabel(r"\bf{Time (s)}")

        # Detection method
        if args.normalised:
            method_plot = method_res_norm
            method_units = r"\bf{ - normalised}"
        else:
            method_plot = method_res
            method_units = r"\bf{ - Calculation Result}"

        fig_method, (mth, thr) = plt.subplots(2,1)
        # # Method result
        mth.plot(method_times, method_plot, label="Signal Processed")
        mth.plot(method_times,detec_pulse, "o", label="Pulse detection")

        if args.normalised:
            mth.plot(method_times, threshold, color="red", alpha = 0.3, label = "CFAR Threshold")
            mth.set_ylim(0, 1.5)
            mth.legend(loc="upper right")

        mth.set_ylabel(method + method_units)

        # # Threshold
        thr.plot(method_times, detect, color="green", label = "Signal detection")
        thr.plot(method_times,detec_pulse, "o", color = "orange", label="Pulse detection")
        thr.set_ylim(0,1.5)
        thr.legend(loc="upper right")
        thr.set_xlabel(r"\bf{Time (s)}")

        # PLT SHOW
        plt.show()
    else: pass

    #######################################################
    # OUTPUT LOG FILE
    #######################################################
    if args.output:
        sys.stdout.close()

    return detect_times


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
    argparser.add_argument("-g", "--goertzel", help="Goertzel method, center frequency, default = 69000", type=int, default=69000)

    argparser.add_argument("-on", "--onFreq", help="Bandpass on frequency in Hz, default 68000 Hz", type=int, default = 68000)
    argparser.add_argument("-off", "--offFreq", help="Bandpass off frequency in Hz, default 70000 Hz", type=int, default=70000)

    argparser.add_argument("-ffreq", "--filterFrequency", help="center signal frequency", type=int, default=69000)
    argparser.add_argument("-LO", "--localOscillator", help="Local Oscillator, default 58000", type=int, default=58000)

    # DETECTION

    argparser.add_argument("-p", "--pulseWidth", help="Pulse Width, default 5 ms", type=int, default=5)

    # OUTPUT
    argparser.add_argument("-sw", "--show", help="show plot default YES (1)", type=int,default=1)
    argparser.add_argument("-n", "--normalised", help="Output plot, 1 Normalised, 0 not normalised", type=int, default=1)
    argparser.add_argument("-o", "--output", help="0 if no output, 1 if log.txt", type=int, default=0)

    # EXIT
    args = argparser.parse_args()

    try:
        times = main(args)
    except KeyboardInterrupt:
        print("Program terminated by user.")
