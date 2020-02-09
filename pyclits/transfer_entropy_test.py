from mutual_inf import renyi_transfer_entropy
from roessler_system import roessler_oscillator
from sample_generator import samples_from_arrays

if __name__ == "__main__":
    kwargs = {"method": "RK45"}
    kwargs["tStop"] = 100

    sol = roessler_oscillator(**kwargs)

    joint_solution = sol.y

    marginal_solution_1 = sol.y[0:3, :].T
    marginal_solution_2 = sol.y[3:6, :].T

    kwargs = {}
    kwargs["transpose"] = True

    samples_x = marginal_solution_1[5:, :]
    samples_marginal_1 = samples_from_arrays(marginal_solution_1, **kwargs)
    samples_marginal_2 = samples_from_arrays(marginal_solution_2, **kwargs)

    results = {}
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
              0.99, 0.999, 1.0, 1.001, 1.01, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65,
              1.7, 1.75, 1.8, 1.85, 1.9, 1.95]
    transfer_entropy = renyi_transfer_entropy(samples_x, samples_marginal_1, samples_marginal_2,
                                              **{"axis_to_join": 1, "method": "LeonenkoProzanto", "alphas": alphas})
    print(transfer_entropy)
