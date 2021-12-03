import os
import pickle
import csv
import numpy as np
import datetime
from operator import itemgetter
from typing import Tuple, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

from sample_generator import shuffle_sample

directory = Path(__file__).parents[0] / "finance_data_reference"
file_pickled = Path(__file__).parents[0] / "finance_data_reference" / "dataset.bin"


def load_tick_data(frame, actual_dataset, actual_metadata):
    price_multiplicator = 10 ** 4
    previous_ask = None
    previous_bid = None
    previous_date = None
    difference_bid = 0
    difference_ask = 0
    difference_date = 0
    for row in frame:
        datetime_str = row[0]
        date = datetime.datetime(int(datetime_str[0:4]), int(datetime_str[4:6]), int(datetime_str[6:8]), int(datetime_str[9:11]), int(datetime_str[11:13]),
                                 int(datetime_str[13:15]), int(datetime_str[15:18]))
        bid = int(float(row[1]) * price_multiplicator)
        ask = int(float(row[2]) * price_multiplicator)
        if previous_ask:
            difference_ask = ask - previous_ask
        if previous_bid:
            difference_bid = bid - previous_bid
        if previous_date:
            difference_date = (date - previous_date).total_seconds()
        previous_ask = ask
        previous_bid = bid
        previous_date = date
        actual_dataset[date] = [bid, ask, difference_bid, difference_ask, difference_date]
    actual_metadata["format"] = "bid, ask, difference_bid, difference_ask, difference_date"
    actual_metadata["price_multiplicator"] = price_multiplicator
    actual_metadata['type'] = 'tick'


def load_aggregated_data(frame, actual_dataset, actual_metadata):
    price_multiplicator = 10 ** 4
    previous_open_price = None
    difference_open_price = 0
    previous_max_price = None
    difference_max_price = 0
    previous_min_price = None
    difference_min_price = 0
    previous_close_price = None
    difference_close_price = 0

    for row in frame:
        datetime_str = row[0].replace('-', '').replace(':', '')
        date = datetime.datetime(int(datetime_str[0:4]), int(datetime_str[4:6]), int(datetime_str[6:8]), int(datetime_str[9:11]), int(datetime_str[11:13]),
                                 int(datetime_str[13:15]))
        open_price = int(float(row[1]) * price_multiplicator)
        max_price = int(float(row[2]) * price_multiplicator)
        min_price = int(float(row[3]) * price_multiplicator)
        close_price = int(float(row[4]) * price_multiplicator)
        volume = (int(row[5]))

        if previous_open_price:
            difference_open_price = open_price - previous_open_price
        previous_open_price = open_price
        if previous_max_price:
            difference_max_price = max_price - previous_max_price
        previous_max_price = max_price
        if previous_min_price:
            difference_min_price = min_price - previous_min_price
        previous_min_price = min_price
        if previous_close_price:
            difference_close_price = close_price - previous_close_price
        previous_close_price = close_price

        actual_dataset[date] = [open_price, max_price, min_price, close_price, difference_open_price, difference_max_price, difference_min_price, difference_close_price, volume]

    actual_metadata['format'] = "open_price, max_price, min_price, close_price, difference_open_price, difference_max_price, difference_min_price, difference_close_price, volume"
    actual_metadata['price_multiplicator'] = price_multiplicator
    actual_metadata['type'] = 'aggregated'


def load_shortened_data(frame, actual_dataset, datafile, actual_metadata):
    year_month = datafile.split("_")[1]
    day = 1
    previous_hour = 0
    previous_price = None
    previous_date = None
    difference_price = 0
    difference_date = 0
    price_multiplicator = 10 ** 5

    for row in frame:
        datetime_str = row[0]
        hour = int(datetime_str[0:2])
        if previous_hour > hour:
            day += 1
        previous_hour = hour

        date = datetime.datetime(int(year_month[0:4]), int(year_month[4:6]), day, int(datetime_str[0:2]), int(datetime_str[2:4]),
                                 int(datetime_str[4:6]), int(datetime_str[6:9]) * 1000)
        price = int(float(row[1]) * price_multiplicator)

        if previous_price:
            difference_price = price - previous_price
        if previous_date:
            difference_date = (date - previous_date).total_seconds()
        previous_price = price
        previous_date = date

        actual_dataset[date] = [price, difference_price, difference_date]
    actual_metadata['format'] = "price, difference_price, difference_date"
    actual_metadata['price_multiplicator'] = price_multiplicator
    actual_metadata['type'] = 'shortened'


def load_datasets():
    dataset = []
    metadata = []
    datafiles = os.listdir(directory)
    datafiles = [file for file in datafiles if "bin" not in file and os.path.isfile(directory / file)]
    for datafile in datafiles:
        actual_dataset = {}
        actual_metadata = {}
        dataset.append(actual_dataset)
        metadata.append(actual_metadata)
        try:
            filename = directory / datafile
            actual_metadata['file'] = datafile
            actual_metadata['directory'] = directory
            actual_metadata['full_filename'] = filename
            actual_metadata['code'] = datafile.split(".")[0].split("_")[0]

            with open(filename) as csvfile:
                # check presences of header
                has_header = csv.Sniffer().has_header(csvfile.read(1024))
                csvfile.seek(0)

                frame = csv.reader(csvfile)
                if has_header:
                    header = next(frame)
                    actual_metadata['header'] = header
                    if len(header) == 2:
                        load_shortened_data(frame, actual_dataset, datafile, actual_metadata)
                    else:
                        load_aggregated_data(frame, actual_dataset, actual_metadata)
                else:
                    load_tick_data(frame, actual_dataset, actual_metadata)
        except EOFError as exc:
            pass

    return dataset, metadata


def prepare_dataset(datasets=None, swap_datasets=False, shuffle_dataset=False, selection1=1, selection2=1):
    filtrated_solution = datasets

    print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} Shape of solution: {filtrated_solution.shape}", flush=True)
    marginal_solution_1 = filtrated_solution[:, 0:selection1]
    marginal_solution_2 = filtrated_solution[:, selection1:selection1+selection2]

    if swap_datasets:
        marginal_solution_1, marginal_solution_2 = (marginal_solution_2, marginal_solution_1)

    if shuffle_dataset:
        marginal_solution_1 = shuffle_sample(marginal_solution_1)

    return marginal_solution_1, marginal_solution_2


def select_dataset_with_code(dataset_with_metadata, code):
    for dataset, metadata in zip(*dataset_with_metadata):
        if metadata['code'] == code:
            return dataset, metadata
    return None


def bid_ask_price_analysis(prefix, dataset, dpi=400, bins=250):
    delta_bids = [record[2] for time, record in dataset.items()]
    delta_asks = [record[3] for time, record in dataset.items()]
    delta_time = [record[4] for time, record in dataset.items()]

    hist, bins_bid, _ = plt.hist(delta_bids, bins=bins)
    plt.yscale('log')
    plt.savefig(prefix+"_bid.png", dpi=dpi)
    plt.close()

    cumsum_bid = hist.cumsum()
    plt.xscale("linear")
    plt.yscale("linear")
    plt.plot(bins_bid[1:], cumsum_bid)
    plt.savefig(prefix+"_bid_cumsum.png", dpi=dpi)
    plt.close()

    hist, bins_ask, _ = plt.hist(delta_asks, bins=bins)
    plt.yscale('log')
    plt.savefig(prefix+"_ask.png", dpi=dpi)
    plt.close()

    cumsum_ask = hist.cumsum()
    plt.xscale("linear")
    plt.yscale("linear")
    plt.plot(bins_ask[1:], cumsum_ask)
    plt.savefig(prefix+"_ask_cumsum.png", dpi=dpi)
    plt.close()

    value_quantile_ask = []
    value_quantile_bid = []
    for quantile in range(1, len(delta_asks)+1):
        index_ask = 0
        for index, item in enumerate(cumsum_ask):
            if item >= quantile:
                index_ask = index
                break

        index_bid = 0
        for index, item in enumerate(cumsum_bid):
            if item >= quantile:
                index_bid = index
                break

        value_quantile_ask.append(bins_ask[index_ask])
        value_quantile_bid.append(bins_bid[index_bid])

    plt.plot(value_quantile_ask, value_quantile_bid)
    plt.savefig(prefix+"_QQ_diagram.png", dpi=dpi)
    plt.close()

    plt.plot(delta_bids, delta_asks, '.')
    plt.savefig(prefix+"_bids_ask_consequent.png", dpi=dpi)
    plt.close()

    corr = signal.correlate(delta_bids, delta_asks, mode='full')
    corr_float = corr / np.sqrt(len(delta_bids) * len(delta_asks))
    plt.plot(corr_float)
    plt.savefig(prefix+"_bids_ask_crosscorrelation.png", dpi=dpi)
    plt.close()

    corr = signal.correlate(np.abs(delta_bids), np.abs(delta_asks), mode='full')
    corr_float = corr / np.max(corr)
    plt.plot(corr_float)
    plt.savefig(prefix+"_bids_ask_abs_crosscorrelation.png", dpi=dpi)
    plt.close()

    plt.plot(delta_bids[0:-1], delta_bids[1:], '.')
    plt.savefig(prefix+"_bids_consequent.png", dpi=dpi)
    plt.close()

    corr = signal.correlate(delta_bids, delta_bids, mode='full')
    corr_float = corr / np.max(corr)
    plt.plot(corr_float)
    plt.savefig(prefix+"_bids_autocorrelation.png", dpi=dpi)
    plt.close()

    corr = signal.correlate(np.abs(delta_bids), np.abs(delta_bids), mode='full')
    corr_float = corr / np.max(corr)
    plt.plot(corr_float)
    plt.savefig(prefix+"_bids_abs_autocorrelation.png", dpi=dpi)
    plt.close()

    corr = signal.correlate(np.sign(delta_bids), np.sign(delta_bids), mode='full')
    corr_float = corr / np.max(corr)
    plt.plot(corr_float)
    plt.savefig(prefix+"_bids_sign_autocorrelation.png", dpi=dpi)
    plt.close()

    plt.plot(delta_asks[0:-1], delta_asks[1:], '.')
    plt.savefig(prefix+"_ask_consequent.png", dpi=dpi)
    plt.close()

    corr = signal.correlate(delta_asks, delta_asks, mode='full')
    corr_float = corr / np.max(corr)
    plt.plot(corr_float)
    plt.savefig(prefix+"_ask_autocorrelation.png", dpi=dpi)
    plt.close()

    corr = signal.correlate(np.abs(delta_asks), np.abs(delta_asks), mode='full')
    corr_float = corr / np.max(corr)
    plt.xscale("linear")
    plt.yscale("linear")
    plt.plot(corr_float)
    plt.savefig(prefix+"_ask_abs_autocorrelation.png", dpi=dpi)
    plt.close()

    corr = signal.correlate(np.sign(delta_asks), np.sign(delta_asks), mode='full')
    corr_float = corr / np.max(corr)
    plt.xscale("linear")
    plt.yscale("linear")
    plt.plot(corr_float)
    plt.savefig(prefix+"_ask_sign_autocorrelation.png", dpi=dpi)
    plt.close()

    hist, bins, = np.histogram(delta_time, bins='auto')
    plt.plot(bins[1:], hist)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(prefix+"_time.png", dpi=dpi)
    plt.close()

    cumsum = hist[::-1].cumsum()[::-1]
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(cumsum)
    plt.savefig(prefix+"_time_cumsum.png", dpi=dpi)
    plt.close()

    logbins = np.logspace(np.log10(bins[1]), np.log10(bins[-1]), 200)
    hist, bins, = np.histogram(delta_time, bins=logbins)
    plt.plot(bins[1:], hist)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(prefix+"_time_log.png", dpi=dpi)
    plt.close()


def interpolation(data_x, data_y, func=lambda x, a, b: a * x + b, selector=lambda x: x < -100, multiplicator=-1, p0=(-1, 11)):
    selector_data = [(np.log(multiplicator * item[0]), np.log(item[1])) for item in zip(data_x, data_y) if selector(item[0]) and item[1] > 0]
    logged_bin = np.array([item[0] for item in selector_data])
    logged_hist = np.array([item[1] for item in selector_data])
    popt, pcov = curve_fit(func, logged_bin, logged_hist, p0=p0)
    return multiplicator * np.exp(logged_bin), np.exp(func(logged_bin, *popt)), popt


def price_analysis(prefix, dataset, dpi=400, number_bins=250):
    delta_price = [record[1] for time, record in dataset.items()]
    delta_time = [record[2] for time, record in dataset.items()]

    hist, bins = np.histogram(delta_price, bins=number_bins)
    interp_bin1, interp_values1, popt1 = interpolation(bins[1:], hist, selector=lambda x: x < -70, multiplicator=-1)
    interp_bin2, interp_values2, popt2 = interpolation(bins[1:], hist, selector=lambda x: x > 70, multiplicator=1)

    dataset_dict = {
        "x": (bins[1:], interp_bin1, interp_bin2), "y": (hist, interp_values1, interp_values2),
        'yscale': 'log', 'legend': True,
        'label': [None, r"$x^{" + f"{popt1[0]:.3f}" + r"}$", r"$x^{" + f"{popt2[0]:.3f}" + r"}$"]
    }
    single_plot(dataset_dict, prefix + "_price.png", dpi=dpi)

    shape = hist.argmax()
    dataset_dict = {
        "x": (bins[:shape+1] * (-1), bins[shape:], -interp_bin1, interp_bin2), "y": (hist[:shape+1], hist[shape-1:], interp_values1, interp_values2),
        'label': ("-", "+", r"$x^{" + f"{popt1[0]:.3f}" + r"}$", r"$x^{" + f"{popt2[0]:.3f}" + r"}$"),
        "xscale": "log", "yscale": "log", 'legend': True,
    }
    single_plot(dataset_dict, prefix + "_price_log_log.png", xlim=(min(bins[:shape+1] * (-1)), max(bins[shape:])), dpi=dpi)

    cumsum_bid = hist.cumsum()
    cfd = cumsum_bid / max(cumsum_bid)
    interp_bin1, interp_values1, popt1 = interpolation(bins[1:], cfd, selector=lambda x: x < -70, multiplicator=-1)
    interp_bin2, interp_values2, popt2 = interpolation(bins[1:], 1 - cfd, selector=lambda x: x > 70, multiplicator=1)
    dataset_dict = {
        "x": (bins[1:], interp_bin1, interp_bin2), "y": (cfd, interp_values1, 1 - interp_values2), 'legend': True,
        'label': (None, r"$x^{" + f"{popt1[0]:.3f}" + r"}$", r"$x^{" + f"{popt2[0]:.3f}" + r"}$"), "yscale": "log"
    }
    single_plot(dataset_dict, prefix + "_price_cumsum.png", dpi=dpi)

    hist, bins = np.histogram(delta_time, bins=number_bins)
    interp_bin, interp_values, popt = interpolation(bins[1:], hist, selector=lambda x: 5 < x < 290, multiplicator=1)
    dataset_dict = {
        "x": (bins[1:], interp_bin), "y": (hist, interp_values), "yscale": 'log', 'legend': True,
        'label': (None, r"$x^{" + f"{popt[0]:.3f}" + r"}$")
    }
    single_plot(dataset_dict, prefix + "_time.png", xlim=(0, max(bins[1:])), dpi=dpi)

    cumsum = hist[::-1].cumsum()[::-1]
    interp_bin, interp_values, popt = interpolation(bins[1:], cumsum, selector=lambda x: 320 < x, multiplicator=1)
    dataset_dict = {
        "x": (bins[1:], interp_bin), "y": (cumsum, interp_values), "xscale": 'log', "yscale": 'log', 'legend': True,
        'label': (None, r"$x^{" + f"{popt[0]:.3f}" + r"}$")
    }
    single_plot(dataset_dict, prefix + "_time_cumsum.png", xlim=(bins[1], bins[-1]), dpi=dpi)

    logbins = np.logspace(np.log10(bins[1]), np.log10(bins[-1]), 200)
    hist, bins = np.histogram(delta_time, bins=logbins)
    interp_bin, interp_values, popt = interpolation(bins[1:], hist, selector=lambda x: x < 280, multiplicator=1)
    dataset_dict = {
        "x": (bins[1:], interp_bin), "y": (hist, interp_values), "xscale": 'log', "yscale": 'log', 'legend': True,
        'label': (None, r"$x^{" + f"{popt[0]:.3f}" + r"}$")
    }
    single_plot(dataset_dict, prefix + "_time_log.png", xlim=(bins[1], bins[-1]), dpi=dpi)


def time_join_dataset(dataset1: Dict[Any, Tuple], dataset2: Dict[Any, Tuple], select_columns1: Tuple, select_columns2: Tuple):
    dataset = []
    set_keys = set()
    keys1 = dataset1.keys()
    set_keys.update(tuple(keys1))

    keys2 = dataset2.keys()
    set_keys.update(tuple(keys2))

    keys = list(set_keys)
    keys.sort()

    getter1 = itemgetter(*select_columns1)
    getter2 = itemgetter(*select_columns2)
    previous_value1 = None
    previous_value2 = None
    for key in keys:
        value1 = dataset1.get(key, None)
        value2 = dataset2.get(key, None)
        if value1 is not None:
            previous_value1 = value1
        if value2 is not None:
            previous_value2 = value2
        if previous_value1 is not None and previous_value2 is not None:
            new_line = []
            new_line.extend(getter1(previous_value1))
            new_line.extend(getter2(previous_value2))
            dataset.append(new_line)
    return np.array(dataset)


def price_minute_analysis(prefix, dataset, dpi=400, bins=250):
    delta_open_price = [record[4] for time, record in dataset.items()]
    delta_max_price = [record[5] for time, record in dataset.items()]
    delta_min_price = [record[6] for time, record in dataset.items()]
    delta_close_price = [record[7] for time, record in dataset.items()]
    volume = [record[8] for time, record in dataset.items()]

    sign_open_price = np.sign(delta_open_price)
    corr = signal.correlate(sign_open_price, sign_open_price, mode='full')

    dataset_dict = {
        "y": corr
    }
    single_plot(dataset_dict, prefix + "_sign_open_autocorrelation.png", dpi=dpi)

    abs_open_price = np.abs(delta_open_price)
    corr = signal.correlate(abs_open_price, abs_open_price, mode='full')
    max_corr = max(corr)
    shape = corr.shape[0] // 2

    dataset_dict = {
        "y": corr[shape:] / max_corr, "xscale": "log", "yscale": "log",
    }
    single_plot(dataset_dict, prefix + "_abs_open_autocorrelation.png", dpi=dpi)

    hist, bins_bid = np.histogram(volume, bins=bins)
    dataset_dict = {
        "x": bins_bid[1:], "y": hist, "xscale": "log", "yscale": "log",
    }
    single_plot(dataset_dict, prefix + "_volume.png", dpi=dpi)

    #logged_bin = np.log(bins_bid[1:])
    #logged_hist = np.log(hist)
    #popt, pcov = curve_fit(lambda x, a, b: a * (x-10**5)**(-b), bins_bid[1:], hist, p0=(10**3, 2))

    cumsum_bid = hist[::-1].cumsum()[::-1]
    dataset_dict = {
        "x": bins_bid[1:], "y": cumsum_bid, "xscale": "log", "yscale": "log",
    }
    single_plot(dataset_dict, prefix + "_volume_cumsum.png", dpi=dpi)

    func = lambda x, a, b: a * x + b
    logged_bin = np.log(bins_bid[1:])
    logged_hist = np.log(cumsum_bid)
    popt, pcov = curve_fit(func, logged_bin, logged_hist, p0=(-1, 2))
    interp_values = np.exp(func(logged_bin, *popt))
    dataset_dict = {
        "x": (bins_bid[1:], bins_bid[1:]), "y": (cumsum_bid, interp_values), "xscale": "log", "yscale": "log", 'label': ["val", r"$x^{" + f"{popt[0]:.3f}" + r"}$"], "legend": True
    }
    single_plot(dataset_dict, prefix + "_volume_cumsum_interp.png", dpi=dpi)

    hist_open, bins_bid_open = np.histogram(delta_open_price, bins=bins)
    hist_max, bins_bid_max = np.histogram(delta_max_price, bins=bins)
    hist_min, bins_bid_min = np.histogram(delta_min_price, bins=bins)
    hist_close, bins_bid_close = np.histogram(delta_close_price, bins=bins)
    dataset_dict = {
        (0, 0): {'x': bins_bid_open[1:], 'y': hist_open, 'title': "open", "yscale": "log"},
        (1, 0): {'x': bins_bid_max[1:], 'y': hist_max, 'title': "max", "yscale": "log"},
        (0, 1): {'x': bins_bid_min[1:], 'y': hist_min, 'title': "min", "yscale": "log"},
        (1, 1): {'x': bins_bid_close[1:], 'y': hist_close, 'title': "close", "yscale": "log"},
    }
    quadruple_plot(dataset_dict, prefix+"_prices.png", dpi=dpi)

    hist_open, bins_bid_open = np.histogram(delta_open_price, bins=bins)
    shape_open = hist_open.argmax()
    hist_max, bins_bid_max = np.histogram(delta_max_price, bins=bins)
    shape_max = hist_max.argmax()
    hist_min, bins_bid_min = np.histogram(delta_min_price, bins=bins)
    shape_min = hist_min.argmax()
    hist_close, bins_bid_close = np.histogram(delta_close_price, bins=bins)
    shape_close = hist_close.argmax()
    dataset_dict = {
        (0, 0): {'x': (bins_bid_open[:shape_open+1] * (-1), bins_bid_open[shape_open:]), 'y': (hist_open[:shape_open+1], hist_open[shape_open-1:]), 'label': ["-", "+"], 'title': "open", "xscale": "log", "yscale": "log"},
        (1, 0): {'x': (bins_bid_max[:shape_max+1] * (-1), bins_bid_max[shape_max:]), 'y': (hist_max[:shape_max+1], hist_max[shape_max-1:]), 'label': ["-", "+"], 'title': "max", "xscale": "log", "yscale": "log"},
        (0, 1): {'x': (bins_bid_min[:shape_min+1] * (-1), bins_bid_min[shape_min:]), 'y': (hist_min[:shape_min+1], hist_min[shape_min-1:]), 'label': ["-", "+"], 'title': "min", "xscale": "log", "yscale": "log"},
        (1, 1): {'x': (bins_bid_close[:shape_close+1] * (-1), bins_bid_close[shape_close:]), 'y': (hist_close[:shape_close+1], hist_close[shape_close-1:]), 'label': ["-", "+"], 'title': "close", "xscale": "log", "yscale": "log"},
    }
    quadruple_plot(dataset_dict, prefix+"_prices_log.png", dpi=dpi)

    dataset_dict = {
        (0, 0): {'x': bins_bid_open[1:], 'y': hist_open.cumsum() / max(hist_open.cumsum()), 'title': "open"},
        (1, 0): {'x': bins_bid_max[1:], 'y': hist_max.cumsum() / max(hist_max.cumsum()), 'title': "max"},
        (0, 1): {'x': bins_bid_min[1:], 'y': hist_min.cumsum() / max(hist_min.cumsum()), 'title': "min"},
        (1, 1): {'x': bins_bid_min[1:], 'y': hist_min.cumsum() / max(hist_min.cumsum()), 'title': "close"},
    }
    quadruple_plot(dataset_dict, prefix+"_price_cumsum.png", dpi=dpi)


def quadruple_plot(datasets, name, dpi=300):
    fig, axs = plt.subplots(2, 2)
    for row in range(2):
        for column in range(2):
            dataset = datasets[(row, column)]
            if isinstance(dataset['x'], (tuple, list)):
                for item_x, item_y, label in zip(dataset['x'], dataset['y'], dataset.get('label', [])):
                    axs[row, column].plot(item_x, item_y, label=label)
            else:
                axs[row, column].plot(dataset['x'], dataset['y'])
            axs[row, column].set_title(dataset.get('title', None))
            axs[row, column].set_xscale(dataset.get('xscale', 'linear'))
            axs[row, column].set_yscale(dataset.get('yscale', 'linear'))

    plt.savefig(name, dpi=dpi)
    plt.close()


def single_plot(dataset, name, xlim=None, ylim=None, dpi=300):
    if 'x' in dataset:
        if isinstance(dataset['x'], (tuple, list)):
            for item_x, item_y, label in zip(dataset['x'], dataset['y'], dataset.get('label', [])):
                plt.plot(item_x, item_y, label=label)
        else:
            plt.plot(dataset['x'], dataset['y'])
    else:
        plt.plot(dataset['y'])

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.title(dataset.get('title', None))
    plt.xscale(dataset.get('xscale', 'linear'))
    plt.yscale(dataset.get('yscale', 'linear'))
    if dataset.get("legend", False):
        plt.legend(ncol=3)
    plt.savefig(name, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    old_data = True
    if old_data:
        with open(file_pickled, "rb") as fh:
            dataset, metadata = pickle.load(fh)
    else:
        dataset, metadata = load_datasets()
        print(f"We aggregated {len(dataset)} records")
        with open(file_pickled, "wb") as fh:
            pickle.dump((dataset, metadata), fh)

    for data, info in zip(dataset, metadata):
        if info['type'] == 'aggregated':
            #price_minute_analysis(info["code"], data, bins=250)
            pass
        elif info['type'] == 'tick':
            #bid_ask_price_analysis(info["code"], data)
            pass
        elif info['type'] == 'shortened':
            price_analysis(info["code"], data)
            pass

    data1, metadata1 = select_dataset_with_code((dataset, metadata), "EURSEK")
    data2, metadata2 = select_dataset_with_code((dataset, metadata), "EURPLN")
    joint_dataset = time_join_dataset(data1, data2, (1, 2), (1, 2))
