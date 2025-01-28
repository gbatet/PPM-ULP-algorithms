# -*- coding: utf-8 -*-
"""

Author: Gerard Batet
Institution: Universitat Politècnica de Catalunya (UPC)
email: gerard.batet@upc.edu
created: 23/01/2024
"""

# IMPORTS

from argparse import ArgumentParser
import matplotlib.pyplot as plt

from demlib.demlib.core import Demodulate as dm

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
    argparser.add_argument("-th", "--threshold", help="Threshold method, threshold value, default = 2 V", type=int, default=2)
    argparser.add_argument("-fft", "--fft", help = "FFT method", action="store_true")
    argparser.add_argument("-goe", "--goertzel", help="Goertzel method, center frequency, default = 69000", type=int, default=69000)
    argparser.add_argument("-fil", "--filter", help = "Filter method", action="store_true")

    # BANDPASS
    argparser.add_argument("-on", "--onFreq", help="Bandpass on frequency in Hz, default 68000 Hz", type=int, default = 68000)
    argparser.add_argument("-off", "--offFreq", help="Bandpass off frequency in Hz, default 70000 Hz", type=int, default=70000)

    # OUTPUT
    argparser.add_argument("-sw", "--show", help="show plot default YES (1)", type=int,default=1)
    argparser.add_argument("-o", "--output", help="output file, default out.txt", type=str, default="out.txt")

    # EXIT
    args = argparser.parse_args()

    # VARIABLES

    try:
        # READ WAV
        sample_rate, samples, time = dm.read_wav() # Samples in counts

        # EMULATE ADC
        # # DownSample
        factor = sample_rate / args.sampleRate
        samples_down = dm.do_downsample_float(samples, factor)

        # # Create buffers
        buffer, buf_times = dm.do_buffers(samples_down, args.buffer)


    # PROCESS BUFFERS
    # CFAR THRESHOLD
    # TOA DETECTION

        # PLOT
        # # ADC input
        fig_gnrl, (gnrl) = plt.subplots()
        gnrl.plot(time, samples)
        gnrl.set_ylabel(r"\bf{ADC Counts - x(n)}")
        gnrl.set_xlabel(r"\bf{Time (s)}")

        # # Detection method

        # # PLT SHOW
        plt.show()



    except KeyboardInterrupt:
        print("Program terminated.")
