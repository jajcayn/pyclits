#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import pandas
import scipy.special as special
import pandas as pd
import processing_datasets
import math
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

if __name__ == "__main__":
    data = pd.read_table('roessler_oscilator/aros2AF.le1', header=None, skiprows=3, delim_whitespace=True)
    processing_datasets.lyapunov_exponent_plot(data, "", """\\LARGE $\\varepsilon$""", """\\LARGE {Lyapunov exponents}""",
                                               ["""\\LARGE$\lambda_1$""", """\\LARGE$\lambda_2$""", """\\LARGE$\lambda_3$""", """\\LARGE$\lambda_4$"""], "lyapunov", "png")
