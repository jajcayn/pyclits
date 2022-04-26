#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import processing_datasets
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    }
)

if __name__ == "__main__":
    gauss = pd.read_csv(
        "roessler_oscilator/escort_distribution/gaus.txt", delimiter=" "
    )
    roesser_x = pd.read_csv(
        "roessler_oscilator/escort_distribution/rossler_x.txt", delimiter=" "
    )
    processing_datasets.escort_distribution(
        [gauss, roesser_x],
        [0.1, 0.5, 2, 10, 1],
        """\\Huge Escort distribution""",
        """\\huge $x$""",
        """\\huge $\\rho_\\alpha$""",
        "escort",
        "png",
    )
