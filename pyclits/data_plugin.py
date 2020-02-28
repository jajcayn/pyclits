import pickle
from pathlib import Path

import numpy as np

file = Path(__file__).parents[0] / "roessler_system_reference" / "arosf11n00eps100raw.dat"
file_pickled = Path(__file__).parents[0] / "roessler_system_reference" / "dataset.bin"


def read_header(fh):
    parameters = {}
    for linenumber in range(6):
        line = fh.readline()
        if "Realization number" in line:
            parameters["Realization number"] = int(line.split("number")[1])
        elif "Ros RAW" in line:
            parameters[line.split(",")[0].replace("#", "")] = line.split(",")[1]
        elif "=" in line:
            if "count" in line:
                value = int(line.split("=")[1])
            else:
                value = float(line.split("=")[1])

            parameters[line.split("=")[0].replace(" ", "").replace("#", "")] = value
        else:
            print(f"Incompatible line detected: {line}, position {fh.tell()}")

    return parameters


def read_dataset(fh, parameter):
    dataset = []
    if "count" in parameter:
        for linenumber in range(parameter["count"]):
            line = fh.readline()
            first = float(line[:15])
            second = float(line[15:])
            dataset.append([first, second])

        frame = np.array(dataset)
        return frame
    else:
        return None


def load_datasets():
    dataset = []
    try:
        with open(file, "rt") as fh:
            while True:
                parameters = read_header(fh)
                frame = read_dataset(fh, parameters)
                if frame is not None:
                    dataset.append([parameters, frame])
                else:
                    break
    except EOFError as exc:
        pass

    return dataset


if __name__ == "__main__":
    dataset = load_datasets()
    print(f"We aggregated {len(dataset)} records")
    with open(file_pickled, "wb") as fh:
        pickle.dump(dataset, fh)
