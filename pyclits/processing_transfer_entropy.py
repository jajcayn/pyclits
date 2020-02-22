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
        # print(old_columns)
        now_columns = [f"transfer_entropy_{item[1]}" for item in old_columns[:-1]]
        column_names = ["alpha"]
        column_names.extend(now_columns)
        column_names.append("epsilon")
        # .append(["epsilon"])
        frame.columns = column_names
        frames.append(frame)

    join_table = pd.concat(frames, ignore_index=True)
    print(join_table)
    pivot_table = pd.pivot_table(join_table, index=['alpha', 'epsilon'])

    pivot_table.to_pickle("transfer_entropy/pivot.bin")
