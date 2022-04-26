#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import pandas
import scipy.special as special
import pandas as pd
import processing_datasets
import math
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    }
)

if __name__ == "__main__":
    column_params = [0.5, 0.9, 1, 1.5, 2, 5]
    data = pd.read_table(
        "qgaussian_distribution/data_q_gauss.txt",
        header=None,
        skiprows=0,
        delim_whitespace=True,
    )
    data.columns = ["X1", "Q-Gauss 1", "Q-Gauss 2", "Q-Gauss 3"]
    data2 = pd.read_table(
        "qgaussian_distribution/data_gauss_escort.txt",
        header=None,
        skiprows=0,
        delim_whitespace=True,
    )
    data2.columns = ["X2", "Gauss escort 1", "Gauss escort 2", "QGauss escort 3"]
    data = pd.concat((data, data2), axis=1)
    processing_datasets.qgauss_plot(
        data,
        """\\Huge Comparison escort distribution of Gaussian distributions and $\\alpha$-Gaussian distribution""",
        """\\Huge $x$""",
        """\\Huge $y$""",
        [
            """\Large $\\alpha$-Gaussian distribution for $\\alpha=0.5$""",
            """\Large $\\alpha$-Gaussian distribution for $\\alpha=1$""",
            """\Large $\\alpha$-Gaussian distribution for $\\alpha=1.5$""",
            """\Large Escort of Gaussian distribution for $\\alpha=0.5$""",
            """\Large Escort of Gaussian distribution for $\\alpha=1$""",
            """\Large Escort of Gaussian distribution for $\\alpha=1.5$""",
        ],
        "qgauss",
        "png",
    )
