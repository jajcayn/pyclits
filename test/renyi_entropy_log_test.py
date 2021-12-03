import math
import numpy as np
import scipy.special as special
import matplotlib
import matplotlib.pyplot as plt


def renyi_entropy(alpha, distances, logarithm=lambda x: np.log(x), order=1, dimension=1, volume=1):
    N = len(distances)
    sum_powers = np.sum(np.power(distances, dimension*(1-alpha)))
    return logarithm(N-1) + logarithm(volume) + (
            logarithm(special.gamma(order))
            - logarithm(special.gamma(order+1-alpha))
            + logarithm(sum_powers/N)) / (1-alpha)


def shannon_entropy(distances, logarithm=lambda x: np.log(x), order=1, dimension=1, volume=1):
    N = len(distances)
    sum_logs = np.sum(logarithm(distances))
    return logarithm(N-1) + logarithm(np.exp(-special.psi(order))) + logarithm(volume) + dimension / N * sum_logs


if __name__ == "__main__":
    volume = 1
    order = 5
    dimension = 1
    distances = np.array([1, 2, 3])
    logarithm = lambda x: np.log10(x)

    alphas_1 = np.linspace(0.1, 1, 50, endpoint=False)
    alphas_2 = np.linspace(1, 3, 100)
    alphas = alphas_1.tolist() + alphas_2.tolist()

    xs=[]
    ys=[]
    for alpha in alphas:
        if alpha == 1:
            entropy = shannon_entropy(distances, logarithm, order, dimension, volume)
        else:
            entropy = renyi_entropy(alpha, distances, logarithm, order, dimension, volume)
        xs.append(alpha)
        ys.append(entropy)

    plt.plot(xs, ys)
    plt.show()
