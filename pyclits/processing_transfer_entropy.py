#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    files = glob.glob("transfer_entropy/Transfer_entropy-*.bin")
    print(files)
    for file in files:
        epsilon = float(file.split("-")[1].split(".b")[0])
        path = Path(file)
        with open(path, "rb") as fb:
            table = pd.read_pickle(fb)

        print(table)
