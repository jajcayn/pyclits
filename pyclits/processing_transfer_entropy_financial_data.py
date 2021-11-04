#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import traceback
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from processing_datasets import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

def load_processed_dataset(dataset, dataset_raw, new_columns_base_name="transfer_entropy_"):
    TE = pd.read_pickle(dataset)
    columns = TE.columns

    TE_raw = pd.read_pickle(dataset_raw)

    return TE, [item for item in TE.columns.tolist() if "mean" in str(item[1])], TE_raw


if __name__ == "__main__":
    dpi = 300
    output = "png"
    directories = ["financial_transfer_entropy_3", "financial_transfer_entropy_2", "financial_transfer_entropy_1", "financial_transfer_entropy"]

    for directory in directories:
        name_of_title = "conditional_information_transfer"
        processed_dataset = directory + "/pivot_dataset.bin"
        processed_raw_dataset = directory + "/pivot_dataset_raw.bin"
        files = glob.glob(processed_dataset)
        if len(files) == 0:
            TE, TE_column_names, TE_raw = process_datasets(
                processed_datasets=directory + "/Conditional_information_transfer-*.bin*",
                result_dataset=processed_dataset,
                result_raw_dataset=processed_raw_dataset,
                take_k_th_nearest_neighbor=5,
                new_columns_base_name=name_of_title,
                converter_epsilon=lambda x: str(x)
            )
        else:
            TE, TE_column_names, TE_raw = load_processed_dataset(processed_dataset, processed_raw_dataset)

        names = {"balance_effective_conditional_information_transfer": 5, "balance_conditional_information_transfer": 4, "effective_conditional_information_transfer": 4, "conditional_information_transfer": 3}

        TE_nonswapped_columns = [item for item in TE_column_names if item[4] == False]

        for item in TE_nonswapped_columns:
            try:
                shift = 0
                for key, value in names.items():
                    if key in item[0]:
                        shift = value
                        break

                item_error = list(item)
                column_name = item[0]
                shuffled_calculation = item[3]
                swapped_datasets = item_error[4]
                item_error[1] = "std"

                history_first_TS = column_name.split("_")[shift]
                history_second_TS = column_name.split("_")[shift+1]
                try:
                    future_first_TS = column_name.split("_")[shift+2]
                except IndexError as err:
                    future_first_TS = None

                name_of_title = column_name.split("r_")[0]+"r"
                balance = "balance" in name_of_title
                latex_title = r"{\Large{" + name_of_title.capitalize().replace("_", " ") + r"}}"
                latex_title_std = latex_title + r"$\large\rm{\ -\ std}$"

                title_graph = {"transfer_entropy": r"$\Large\rm{Transfer\ entropy}$",
                               "conditional_information_transfer": r"$\Large\rm{Conditional\ information\ transfer}$", }
                filename_direction = {True: "Y->X", False: "X->Y"}
                title_map = {(False, False): r"{\alpha: X\rightarrow Y}", (True, False): r"{\alpha: X_{shuffled}\rightarrow Y}",
                             (False, True): r"{\alpha: Y\rightarrow X}", (True, True): r"{\alpha: Y_{shuffled}\rightarrow X}"}

                if future_first_TS is not None:
                    if balance:
                        label = "$T^{}_{} ([{}],[{}],[{}])$".format("{(R, eff)}" if "effective" in column_name else "{(R)}",
                                                                    title_map[(shuffled_calculation, swapped_datasets)], history_first_TS,
                                                                    history_second_TS, future_first_TS) + "-" + \
                                "$T^{}_{} ([{}],[{}],[{}])$".format("{(R, eff)}" if "effective" in column_name else "{(R)}",
                                                                    title_map[(shuffled_calculation, not swapped_datasets)], history_first_TS,
                                                                    history_second_TS, future_first_TS)
                    else:
                        label = "$T^{}_{} ([{}],[{}],[{}])$".format("{(R, eff)}" if "effective" in column_name else "{(R)}",
                                                                    title_map[(shuffled_calculation, swapped_datasets)],
                                                                    history_first_TS, history_second_TS, future_first_TS)
                else:
                    label = "$T^{}_{} ([{}],[{}])$".format("{(R, eff)}" if "effective" in column_name else "{(R)}",
                                                           title_map[(shuffled_calculation, swapped_datasets)], history_first_TS, history_second_TS)

                print(column_name, label)

                errorbar_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_2d_bars"
                standard_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_2d"
                plot_3D_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "")
                plot_3D_surf_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled_surf" if shuffled_calculation else "_surf")
                plot_2D_filename_implot = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_implot"
                plot_2D_filename_implot_std = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_implot_std"
                std_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_2d_std"

                figures2d_TE_alpha(TE, item, latex_title, r"$\alpha$", label, standard_filename, output, dpi=dpi)
                figures2d_TE_alpha(TE, tuple(item_error), latex_title_std, r"$\alpha$", label, std_filename, output, dpi=dpi)
                figures2d_TE_alpha_errorbar(TE, item, tuple(item_error), latex_title, r"$\alpha$", label, errorbar_filename, output, dpi=dpi)
            except Exception as exc:
                print(f"Problem {exc} {item}")
                traceback.print_exc()

        del TE, TE_column_names, TE_raw

    print("Finished")
