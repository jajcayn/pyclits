from mutual_inf import renyi_transfer_entropy
from roessler_system import roessler_oscillator
from sample_generator import samples_from_arrays

if __name__ == "__main__":
    kwargs = {"method": "RK45"}
    kwargs["tStop"] = 100

    sol = roessler_oscillator(**kwargs)

    joint_solution = sol.y
    # samples_joint = samples_from_arrays(joint_solution)
    marginal_solution_1 = sol.y[0:3, :].T
    marginal_solution_2 = sol.y[3:6, :].T

    kwargs = {}
    kwargs["transpose"] = True

    samples_x = samples_from_arrays(marginal_solution_1, **kwargs)
    samples_marginal_1 = samples_from_arrays(marginal_solution_1, **kwargs)
    samples_marginal_2 = samples_from_arrays(marginal_solution_2, **kwargs)

    transfer_entropy = renyi_transfer_entropy(marginal_solution_1, samples_marginal_1, samples_marginal_2,
                                              **{"axis_to_join": 1})
