import time

from mutual_inf import renyi_transfer_entropy
from roessler_system import roessler_oscillator
from sample_generator import preparation_dataset_for_transfer_entropy

if __name__ == "__main__":
    configuration = {"method": "LSODA", "tInc": 0.001, "cache": True}
    configuration["tStop"] = 10000

    t0 = time.process_time()
    sol = roessler_oscillator(**configuration)
    t1 = time.process_time()
    duration = t1 - t0
    print(f"Solution duration [s]: {duration}")

    # preparation of sources
    filtrated_solution = sol.y
    joint_solution = filtrated_solution
    marginal_solution_1 = filtrated_solution[0:3, :].T
    marginal_solution_2 = filtrated_solution[3:6, :].T

    histories = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
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
        alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                  0.9, 0.95,
                  0.99, 0.999, 1.0, 1.001, 1.01, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65,
                  1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 1.99]
        indices_to_use = list(range(50))
        configuration = {"transpose": True, "axis_to_join": 0, "method": "LeonenkoProzanto", "alphas": alphas,
                         "enhanced_calculation": True, "indices_to_use": indices_to_use}
        t0 = time.process_time()
        transfer_entropy = renyi_transfer_entropy(y, y_hist, z, **configuration)
        t1 = time.process_time()
        duration = t1 - t0
        print(f"Calculation of transfer entropy [s]: {duration}")
        print(f"Transfer Renyi entropy: {transfer_entropy}")
