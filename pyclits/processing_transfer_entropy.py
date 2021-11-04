#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
from collections import Counter
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
    dpi = 150
    output = "png"
    #"roessler_oscilator/conditional_information_transfer_Dh=1",
    #"roessler_oscilator/conditional_information_transfer_Dh=2",
    #"roessler_oscilator/conditional_information_transfer_Dh=n",
    #"roessler_oscilator/conditional_information_transfer_full_Dh=1",
    #"roessler_oscilator/conditional_information_transfer_full_Dh=2",
    #"roessler_oscilator/conditional_information_transfer_full_Dh=n",
    #"conditional_information_transfer_GARCH_single",
    #"roessler_oscilator/conditional_information_transfer_X_3_Y_3",
    #"roessler_oscilator/conditional_information_transfer_X_3_Y_1",
    #"roessler_oscilator/conditional_information_transfer_X_1_Y_3",

    directories = [
        "roessler_oscilator/conditional_information_transfer_X_2_Y_2"]
    #directory = "transfer_entropy"

    for directory in directories:
        name_of_title = "conditional_information_transfer"
        processed_dataset = directory + "/pivot_dataset.bin"
        processed_raw_dataset = directory + "/pivot_dataset_raw.bin"
        files = glob.glob(processed_dataset)
        if len(files) == 0:
            TE, TE_column_names, TE_raw = process_datasets(processed_datasets=directory + "/Conditional_information_transfer-*.bin",
                                                           result_dataset=processed_dataset, result_raw_dataset=processed_raw_dataset,
                                                           new_columns_base_name=name_of_title)
        else:
            TE, TE_column_names, TE_raw = load_processed_dataset(processed_dataset, processed_raw_dataset)

        names = {"balance_effective_conditional_information_transfer": 5, "balance_conditional_information_transfer": 4, "effective_conditional_information_transfer": 4, "conditional_information_transfer": 3}
        for item in TE_column_names:
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
                latex_title = None
                latex_title_std = None
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

                latex_epsilon_label = r"$\varepsilon$"

                figures2d_imshow(TE, item, latex_title, label, plot_2D_filename_implot, suffix=output, cmap="rainbow", dpi=dpi)
                figures2d_imshow(TE, tuple(item_error), latex_title, label, plot_2D_filename_implot_std, suffix=output, cmap="rainbow", dpi=dpi)
                figures3d_surface_TE(TE, item, latex_title, label, plot_3D_surf_filename, suffix=output, cmap="rainbow", dpi=dpi)
                figures2d_TE_errorbar(TE, item, tuple(item_error), latex_title, latex_epsilon_label, label, errorbar_filename, suffix=output, dpi=dpi)
                figures2d_TE(TE, item, latex_title, latex_epsilon_label, label, standard_filename, suffix=output, dpi=dpi)
                figures3d_TE(TE, item, latex_title, label, plot_3D_filename, suffix=output, dpi=dpi)
                figures2d_TE(TE, tuple(item_error), latex_epsilon_label, latex_title_std, label, std_filename, suffix=output, dpi=dpi)
            except Exception as exc:
                print(f"Problem {exc} {item}")
                traceback.print_exc()

        del TE, TE_column_names, TE_raw

    print("Finished")
