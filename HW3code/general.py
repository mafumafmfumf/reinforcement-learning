import time
import sys
import logging
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def export_plot(ys, ylabel, filename):
    """
    Export a plot in filename

    Args:
        ys: (list) of float / int to plot
        filename: (string) directory
    """
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()