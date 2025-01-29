# -*- coding: utf-8 -*-
"""

Author: Gerard Batet
Institution: Universitat Politècnica de Catalunya (UPC)
email: gerard.batet@upc.edu
created: 23/01/2024
"""

# IMPORTS

import demlib as dm

from argparse import ArgumentParser
import matplotlib.pyplot as plt



plt.rcParams.update({
    "text.usetex": True
})

# MAIN

if __name__ == "__main__":

    # INIT
    argparser = ArgumentParser()

    # INFO
    argparser.add_argument("-v", "--verbose", help="more info", action="store_true")

    # INPUT
    argparser.add_argument("-i", "--input", help="Input file directory and name, default docs/4.wav", type=str, default="docs/4.wav")

    # ACQUISITION
    argparser.add_argument("-sr", "--sampleRate", help="Sample Rate of the uC, default 150000 S/s", type=int, default=150000)
    argparser.add_argument("-b", "--buffer", help="ADC Buffer lenght, default = 256", type=int, default=256)

    # PROCESSING
    argparser.add_argument("-m", "--method", help="Processing method, 1: FFT, 2: Go, 3: Filt, default = 1", type = int, default=1)

    # CONFIG METHODS
    argparser.add_argument("-goe", "--goertzel", help="Goertzel method, center frequency, default = 69000", type=int, default=69000)
    argparser.add_argument("-on", "--onFreq", help="Bandpass on frequency in Hz, default 68000 Hz", type=int, default = 68000)
    argparser.add_argument("-off", "--offFreq", help="Bandpass off frequency in Hz, default 70000 Hz", type=int, default=70000)

    # OUTPUT
    argparser.add_argument("-sw", "--show", help="show plot default YES (1)", type=int,default=1)
    argparser.add_argument("-o", "--output", help="output file, default out.txt", type=str, default="out.txt")

    # EXIT
    args = argparser.parse_args()

    ############
    # VARIABLES
    ############
    method = ""         # Variable for method of processing
    method_units = ""   # Variable for method units

    try:
        #######################################################
        # READ DATA
        #######################################################
        sample_rate, samples, time = dm.read_wav(args.input) # Samples in counts

        #######################################################
        # EMULATE ADC OPERATIVE
        #######################################################

        # DownSample
        factor = sample_rate / args.sampleRate
        samples_down = dm.do_downsample_float(samples, factor)

        # Create buffers
        buffer, buf_times = dm.do_buffers(samples_down, args.buffer, args.sampleRate)

        #######################################################
        # PROCESS THE BUFFERS
        #######################################################

        # FFT
        if args.method == 1:
            method_units = r"\bf{Spectral Power}"
            fft_res, lb, ub = dm.do_fft(buffer, args.sampleRate, args.onFreq, args.offFreq)

            method = r"\bf{FFT}"
            print("Method: FFT")
            print(f"Set frequency bandpass: {args.onFreq} -{args.offFreq} Hz ")
            print(f"Real frequency bandpass: {lb} -{ub} Hz ")

        # Goertzel
        elif args.method == 2:
            pass

        # Filtering
        elif args.method == 3:
            pass

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
        mth.plot(buf_times, fft_res)
        mth.set_ylabel(method_units)
        mth.set_xlabel(r"\bf{Time (s)}")


        # # PLT SHOW
        plt.show()



    except KeyboardInterrupt:
        print("Program terminated.")
