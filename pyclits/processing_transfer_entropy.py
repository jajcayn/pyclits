#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    files = glob.glob("transfer_entropy/Transfer_entropy-*.bin")
    print(files)
    frames = []
    for file in files:
        epsilon = float(file.split("-")[1].split(".b")[0])
        path = Path(file)

        table = pd.read_pickle(path)

        frame = pd.DataFrame(table)
        frame["epsilon"] = epsilon
        old_columns = frame.columns
        frame = frame.reset_index()
        frame.columns = ["alpha", f"transfer_entropy_{old_columns[0]}", "epsilon"]
        frames.append(frame)

    join_table = pd.concat(frames, ignore_index=True)
    print(join_table)
    print(pd.pivot_table(join_table, index=['alpha', 'epsilon']))
