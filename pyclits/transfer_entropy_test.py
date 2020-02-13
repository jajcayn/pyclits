import time
from pathlib import Path

import numpy as np

from mutual_inf import renyi_transfer_entropy
from roessler_system import roessler_oscillator
from sample_generator import preparation_dataset_for_transfer_entropy

if __name__ == "__main__":
    epsilons = [0.1]
    for espilon in epsilons:
        configuration = {"method": "LSODA", "tInc": 0.001, "cache": True, "epsilon": epsilon}
        configuration["tStop"] = 10000

        t0 = time.process_time()
        sol = roessler_oscillator(**configuration)
        t1 = time.process_time()
        duration = t1 - t0
        print(f"Solution duration [s]: {duration}")

        # preparation of sources
        filtrated_solution = sol.y[:, 2000:]
        joint_solution = filtrated_solution
        marginal_solution_1 = filtrated_solution[0:3, :].T
        marginal_solution_2 = filtrated_solution[3:6, :].T

        alphas = np.linspace(0.19, 1.99, 199)
        results = {}
        histories = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17, 20, 22, 25, 27, 30, 50]
        for history in histories:
            solution_size = joint_solution.shape
            configuration = {"transpose": True, "history_x": history, "history_y": history}

            t0 = time.process_time()
            y, y_hist, z = preparation_dataset_for_transfer_entropy(marginal_solution_1, marginal_solution_2,
                                                                    **configuration)
            t1 = time.process_time()
            duration = t1 - t0
            print(f"Preparation of datasets [s]: {duration}")

            results = {}
            indices_to_use = list(range(1, 30))
            configuration = {"transpose": True, "axis_to_join": 0, "method": "LeonenkoProzanto", "alphas": alphas,
                             "enhanced_calculation": True, "indices_to_use": indices_to_use}
            t0 = time.process_time()
            transfer_entropy = renyi_transfer_entropy(y, y_hist, z, **configuration)
            t1 = time.process_time()
            duration = t1 - t0
            print(f"Calculation of transfer entropy [s]: {duration}")
            print(f"Transfer Renyi entropy with {history}: {transfer_entropy}")

            results[history] = transfer_entropy

        path = Path(f"trasfer_entropy/Transfer_entropy-{espilon}.bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fb:
            import pickle

            pickle.dump(results, fb)
