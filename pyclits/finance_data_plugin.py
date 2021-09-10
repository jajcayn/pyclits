import os
import pickle
import csv
import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

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


def price_analysis(prefix, dataset, dpi=400, bins=250):
    delta_price = [record[1] for time, record in dataset.items()]
    delta_time = [record[2] for time, record in dataset.items()]

    hist, bins_bid = np.histogram(delta_price, bins=bins)
    plt.plot(bins_bid[1:], hist)
    plt.yscale('log')
    plt.savefig(prefix+"_price.png", dpi=dpi)
    plt.close()

    shape = hist.argmax()
    plt.plot(bins_bid[:shape+1] * (-1), hist[:shape+1], label="-")
    plt.plot(bins_bid[shape:], hist[shape-1:], label="+")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(prefix+"_price_log_log.png", dpi=dpi)
    plt.close()

    cumsum_bid = hist.cumsum()
    plt.plot(bins_bid[1:], cumsum_bid)
    plt.xscale("linear")
    plt.yscale("linear")
    plt.savefig(prefix+"_price_cumsum.png", dpi=dpi)
    plt.close()

    hist, bins = np.histogram(delta_time, bins=bins)
    plt.plot(bins[1:], hist)
    plt.yscale('log')
    plt.xscale('linear')
    plt.savefig(prefix+"_time.png", dpi=dpi)
    plt.close()

    cumsum = hist[::-1].cumsum()[::-1]
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(cumsum)
    plt.savefig(prefix+"_time_cumsum.png", dpi=dpi)
    plt.close()

    logbins = np.logspace(np.log10(bins[1]), np.log10(bins[-1]), 200)
    hist, bins = np.histogram(delta_time, bins=logbins)
    plt.plot(bins[1:], hist)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(prefix+"_time_log.png", dpi=dpi)
    plt.close()


def price_minute_analysis(prefix, dataset, dpi=400, bins=250):
    delta_open_price = [record[4] for time, record in dataset.items()]
    delta_max_price = [record[5] for time, record in dataset.items()]
    delta_min_price = [record[6] for time, record in dataset.items()]
    delta_close_price = [record[7] for time, record in dataset.items()]
    volume = [record[8] for time, record in dataset.items()]

    sign_open_price = np.sign(delta_open_price)
    corr = signal.correlate(sign_open_price, sign_open_price, mode='full')

    plt.xscale("linear")
    plt.yscale("linear")
    plt.plot(corr)
    plt.savefig(prefix+"_sign_open_autocorrelation.png", dpi=dpi)
    plt.close()

    abs_open_price = np.abs(delta_open_price)
    corr = signal.correlate(abs_open_price, abs_open_price, mode='full')
    max_corr = max(corr)
    shape = corr.shape[0] // 2

    plt.xscale("log")
    plt.yscale("log")
    plt.plot(corr[shape:] / max_corr)
    plt.savefig(prefix+"_abs_open_autocorrelation.png", dpi=dpi)
    plt.close()

    hist, bins_bid = np.histogram(volume, bins=bins)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(bins_bid[1:], hist)
    plt.savefig(prefix+"_volume.png", dpi=dpi)
    plt.close()

    cumsum_bid = hist.cumsum()
    plt.plot(bins_bid[1:], cumsum_bid)
    plt.xscale("log")
    plt.yscale("linear")
    plt.savefig(prefix+"_volume_cumsum.png", dpi=dpi)
    plt.close()

    fig, axs = plt.subplots(2, 2)
    hist_open, bins_bid_open = np.histogram(delta_open_price, bins=bins)
    axs[0, 0].plot(bins_bid_open[1:], hist_open)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('open')
    hist_max, bins_bid_max = np.histogram(delta_max_price, bins=bins)
    axs[1, 0].plot(bins_bid_max[1:], hist_max)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('max')
    hist_min, bins_bid_min = np.histogram(delta_min_price, bins=bins)
    axs[0, 1].plot(bins_bid_min[1:], hist_min)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('min')
    hist_close, bins_bid_close = np.histogram(delta_close_price, bins=bins)
    axs[1, 1].plot(bins_bid_close[1:], hist_close)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('close')
    fig.savefig(prefix + "_prices.png", dpi=dpi)
    plt.close()

    fig, axs = plt.subplots(2, 2)
    hist_open, bins_bid_open = np.histogram(delta_open_price, bins=bins)
    shape = hist_open.argmax()
    axs[0, 0].plot(bins_bid_open[:shape+1] * (-1), hist_open[:shape+1], label="-")
    axs[0, 0].plot(bins_bid_open[shape:], hist_open[shape-1:], label="+")
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('open')
    axs[0, 0].legend()
    hist_max, bins_bid_max = np.histogram(delta_max_price, bins=bins)
    shape = hist_max.argmax()
    axs[1, 0].plot(bins_bid_max[:shape+1] * (-1), hist_max[:shape+1], label="-")
    axs[1, 0].plot(bins_bid_max[shape:], hist_max[shape-1:], label="+")
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('max')
    axs[1, 0].legend()
    hist_min, bins_bid_min = np.histogram(delta_min_price, bins=bins)
    shape = hist_min.argmax()
    axs[0, 1].plot(bins_bid_min[:shape+1] * (-1), hist_min[:shape+1], label="-")
    axs[0, 1].plot(bins_bid_min[shape:], hist_min[shape-1:], label="+")
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('min')
    axs[0, 1].legend()
    hist_close, bins_bid_close = np.histogram(delta_close_price, bins=bins)
    shape = hist_close.argmax()
    axs[1, 1].plot(bins_bid_close[:shape+1] * (-1), hist_close[:shape+1], label="-")
    axs[1, 1].plot(bins_bid_close[shape:], hist_close[shape-1:], label="+")
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('close')
    axs[1, 1].legend()
    fig.savefig(prefix + "_prices_log.png", dpi=dpi)
    plt.close()

    fig, axs = plt.subplots(2, 2)
    hist_open_cumsum = hist_open.cumsum()
    axs[0, 0].plot(bins_bid_open[1:], hist_open_cumsum)
    axs[0, 0].set_title('open')
    hist_max_cumsum = hist_max.cumsum()
    axs[1, 0].plot(bins_bid_max[1:], hist_max_cumsum)
    axs[1, 0].set_title('max')
    hist_min_cumsum = hist_min.cumsum()
    axs[0, 1].plot(bins_bid_min[1:], hist_min_cumsum)
    axs[0, 1].set_title('min')
    hist_close_cumsum = hist_close.cumsum()
    axs[1, 1].plot(bins_bid_close[1:], hist_close_cumsum)
    axs[1, 1].set_title('close')
    plt.savefig(prefix+"_price_cumsum.png", dpi=dpi)
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
            price_minute_analysis(info["code"], data, bins=250)
            pass
        elif info['type'] == 'tick':
            bid_ask_price_analysis(info["code"], data)
            pass
        elif info['type'] == 'shortened':
            price_analysis(info["code"], data)
            pass
