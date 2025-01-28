# IMPORTS
from argparse import ArgumentParser

import pandas as pd
import pyarrow

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True
})

#MAIN
if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("-v", "--verbose", help="more info", action="store_true")

    argparser.add_argument("-i", "--input", help="Input file directory and name, default csv/analog.csv", type=str, default="csv/analog.csv")
    argparser.add_argument("-x", "--xdata", help="Input column name for x, default Time(s)", type=str, default="Time(s)")
    argparser.add_argument("-y", "--ydata", help="Input column name for y default Data", type=str, default="Data")

    args = argparser.parse_args()


    # CSV READ
    df = pd.read_csv(args.input)

   # time = df[args.xdata]
    data = df[args.ydata]

    #FIGURE
    fig, ax = plt.subplots()

    ax.plot(data)

    plt.show()