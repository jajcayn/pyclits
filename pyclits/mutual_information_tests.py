
import numpy as np
import scipy.special as spec
import time
import mutual_inf
import math

time_start = time.process_time()


def Renyi_normal_distribution(sigma, alpha):
    if isinstance(sigma, float):
        return Renyi_normal_distribution_1D(sigma, alpha)
    elif isinstance(sigma, np.matrix):
        return Renyi_normal_distribution_ND(sigma, alpha)
    else:
        raise ArithmeticError("sigma parameter has wrong type")


def Renyi_normal_distribution_1D(sigma_number, alpha):
    if alpha == 1:
        return math.log(2*math.pi*math.exp(1)*np.power(sigma_number, 2))/2
    else:
        return math.log2(2*math.pi) / 2 + math.log2(sigma) + math.log2(alpha) / (1 - alpha) / 2


def Renyi_normal_distribution_ND(sigma_matrix: np.matrix, alpha):
    return math.log2(2*math.pi) / 2 + math.log2(np.linalg.det(sigma_matrix)) + math.log2(alpha) / (1 - alpha) / 2


def Renyi_student_t_distribution(sigma, degrees_of_freedom, alpha):
    if len(sigma.shape) == 2 and (sigma.shape[0] == sigma.shape[1]):
        dimension = sigma.shape[0]
        beta_factor = math.log2(spec.beta(dimension/2.0, alpha*(dimension+degrees_of_freedom)/2.0 - dimension/2.0)/ spec.beta(degrees_of_freedom/2.0, math.pow(dimension/2.0), alpha))
        return 1 / (1 - alpha) * beta_factor * math.log2(math.pow(np.pi*degrees_of_freedom, dimension)*np.linalg.det(sigma)) - math.log2(spec.gamma(dimension/2.0))
    else:
        raise ArithmeticError("sigma parameter has wrong type")

def Renyi_beta_distribution(a, b, alpha):
    return 1 / (1 - alpha) * math.log2(spec.beta(alpha*a+alpha-1, alpha*b+alpha-1) / math.pow(spec.beta(a, b), alpha))

if __name__ == "__main__":
    import time

    sample_array = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=float)
    input_sample = np.ndarray(shape=sample_array.shape, buffer=sample_array)
    #print(input_sample)
    print(mutual_inf.renyi_entropy(np.matrix([[1],[2],[3],[4],[5],[6],[7],[8],[9]]), method="LeonenkoProzanto"))
    print(mutual_inf.renyi_entropy(input_sample, method="Paly"))

    mu = 0
    sigma = 10

    number_samples = 100
    alpha = 0.98
    alphas = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.9]
    for alpha in alphas:
        for number_samples in [10, 20, 50, 100, 200, 500, 1000, 2000]:
            entropy = 0
            samples = np.random.normal(mu, sigma, (number_samples, 1))
            time_start = time.process_time()
            #entropy = mutual_inf.renyi_entropy(samples, method="Lavicka", indices_to_use=[1])
            time_end = time.process_time()
            #print(number_samples, time_end-time_start, entropy, Renyi_normal_distribution_1D(sigma, alpha))

            time_start = time.process_time()
            entropy = mutual_inf.renyi_entropy(samples, method="LeonenkoProzanto", indices_to_use=[1, 2, 3, 4], alpha=alpha)
            time_end = time.process_time()
            print(f"samples={number_samples}, duration={time_end-time_start}, alpha={alpha}, tested_estimator={entropy}, theoretical_calculation={Renyi_normal_distribution_1D(sigma, alpha)}")
