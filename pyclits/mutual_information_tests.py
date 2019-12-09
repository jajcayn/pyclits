
import numpy as np
import scipy.special as spec
import time
import mutual_inf
import math
import sys

time_start = time.process_time()


def sample_normal_distribution(sigma, size_sample):
    if isinstance(sigma, np.ndarray) and len(sigma.shape) == 2 and (sigma.shape[0] == sigma.shape[1]):
        dimension = sigma.shape[0]
        uncorrelated_sample = np.random.normal(0, 1.0, (dimension, size_sample))
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        standard_deviations = np.sqrt(eigenvalues)
        identity_sqrt = np.diag(standard_deviations)
        scaled_sample = identity_sqrt.dot(uncorrelated_sample)
        correlated_sample = eigenvectors.dot(scaled_sample)

        return correlated_sample.T
    else:
        raise ArithmeticError("sigma parameter has wrong type")


def Renyi_normal_distribution(sigma, alpha):
    if isinstance(sigma, float):
        return Renyi_normal_distribution_1D(sigma, alpha)
    elif isinstance(sigma, np.matrix):
        return Renyi_normal_distribution_ND(sigma, alpha)
    else:
        raise ArithmeticError("sigma parameter has wrong type")


def Renyi_normal_distribution_1D(sigma_number, alpha):
    if alpha == 1:
        return math.log2(2 * math.pi * math.exp(1) * np.power(sigma_number, 2))/2
    else:
        return math.log2(2 * math.pi) / 2 + math.log2(sigma_number) + math.log2(alpha) / (alpha - 1) / 2


def Renyi_normal_distribution_ND(sigma_matrix: np.matrix, alpha):
    dimension = sigma_matrix.shape[0]
    if alpha == 1:
        return math.log2(2*math.pi*math.exp(1)) * dimension/2.0 + math.log2(math.sqrt(np.linalg.det(sigma_matrix)))
    else:
        return math.log2(2*math.pi) * dimension / 2 + math.log2(np.linalg.det(sigma_matrix)) / 2.0 + dimension * math.log2(alpha) / (alpha - 1) / 2


def Renyi_student_t_distribution_1D(sigma, degrees_of_freedom, alpha):
    if isinstance(sigma, float) or isinstance(sigma, int):
        dimension = 1
        determinant = sigma
    elif isinstance(sigma, np.matrix):
        if len(sigma.shape) == 2 and (sigma.shape[0] == sigma.shape[1]):
            dimension = sigma.shape[0]
            determinant = np.linalg.det(sigma)
        else:
            raise ArithmeticError("sigma parameter has wrong type")
    else:
        raise ArithmeticError("sigma parameter has wrong type")

    if alpha == 1:
        return (degrees_of_freedom+1.0)/2*(spec.digamma((degrees_of_freedom+1.0)/2) - spec.digamma((degrees_of_freedom)/2))+np.log2(np.sqrt(degrees_of_freedom) * spec.beta(0.5, degrees_of_freedom/2.0))
    else:
        nominator = spec.beta(dimension/2.0, alpha*(dimension+degrees_of_freedom)/2.0 - dimension/2.0)
        denominator = math.pow(spec.beta(degrees_of_freedom / 2.0, dimension / 2.0), alpha)
        beta_factor = math.log2(nominator / denominator)
        return 1 / (1 - alpha) * beta_factor * math.log2(math.pow(np.pi*degrees_of_freedom, dimension)*determinant) - math.log2(spec.gamma(dimension/2.0))

def Renyi_beta_distribution(a, b, alpha):
    return 1 / (1 - alpha) * math.log2(spec.beta(alpha*a+alpha-1, alpha*b+alpha-1) / math.pow(spec.beta(a, b), alpha))

def complete_test_1D(filename="statistics.txt", samples = 1000,
                  alphas=[0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9],
                  mu = 0, sigmas = [0.1, 0.5, 1.0, 5.0, 10.0, 50, 100],
                  sizes_of_sample = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
                  theoretical_value_function=lambda sigma, alpha: Renyi_normal_distribution_1D(sigma, alpha),
                  sample_generator=lambda mu, sigma, size_sample: np.random.normal(mu, sigma, (size_sample, 1)),
                  sample_estimator=lambda data_samples, alpha: mutual_inf.renyi_entropy(data_samples, method="LeonenkoProzanto", indices_to_use=[1, 2, 3, 4, 5], alpha=alpha)):
    with open(filename, "wt") as fd:
        entropy_samples = {}
        duration_samples = {}
        difference_samples = {}

        for sigma in sigmas:
            for alpha in alphas:
                for size_sample in sizes_of_sample:
                    sample_position = (alpha, size_sample, sigma)
                    theoretical_value = theoretical_value_function(sigma, alpha)

                    entropy_samples[sample_position] = []
                    duration_samples[sample_position] = []
                    difference_samples[sample_position] = []

                    for sample in range(1, samples + 1):
                        if sample % 10 == 0:
                            print(f"alpha = {alpha}, size_sample = {size_sample}, sigma={sigma}, sample = {sample}")

                        data_samples = sample_generator(mu, sigma, size_sample)

                        time_start = time.process_time()
                        entropy = sample_estimator(data_samples, alpha=alpha)
                        time_end = time.process_time()

                        duration = time_end - time_start
                        difference = theoretical_value - entropy

                        entropy_samples[sample_position].append(entropy)
                        duration_samples[sample_position].append(duration)
                        difference_samples[sample_position].append(difference)

                        #print(f"samples={size_sample}, duration={time_end-time_start}, alpha={alpha}, tested_estimator={entropy}, theoretical_calculation={theoretical_value}, difference={difference}")

                    print(f"{alpha} {size_sample} {sigma} {np.mean(entropy_samples[sample_position])} {np.std(entropy_samples[sample_position])} {np.mean(duration_samples[sample_position])} {np.std(duration_samples[sample_position])} {np.mean(difference_samples[sample_position])} {np.std(difference_samples[sample_position])}", file=fd)

def complete_test_ND(filename="statistics.txt", samples = 1000, sigma_skeleton = np.identity(10),
                  alphas=[0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9],
                  mu = 0, sigmas = [0.1, 0.5, 1.0, 5.0, 10.0, 50, 100],
                  sizes_of_sample = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
                  theoretical_value_function=lambda sigma, alpha: Renyi_normal_distribution_ND(sigma, alpha),
                  sample_generator=lambda mu, sigma, size_sample: sample_normal_distribution(sigma, size_sample),
                  sample_estimator=lambda data_samples, alpha: mutual_inf.renyi_entropy(data_samples, method="LeonenkoProzanto", indices_to_use=[1, 2, 3, 4, 5], alpha=alpha)):
    with open(filename, "wt") as fd:
        entropy_samples = {}
        duration_samples = {}
        difference_samples = {}

        for sigma in sigmas:
            matrix_sigma = sigma * sigma_skeleton
            for alpha in alphas:
                for size_sample in sizes_of_sample:
                    sample_position = (alpha, size_sample, sigma)
                    theoretical_value = theoretical_value_function(matrix_sigma, alpha)

                    entropy_samples[sample_position] = []
                    duration_samples[sample_position] = []
                    difference_samples[sample_position] = []

                    for sample in range(1, samples + 1):
                        if sample % 10 == 0:
                            print(f"alpha = {alpha}, size_sample = {size_sample}, sigma={sigma}, sample = {sample}")

                        data_samples = sample_generator(mu, matrix_sigma, size_sample)

                        time_start = time.process_time()
                        entropy = sample_estimator(data_samples, alpha=alpha)
                        time_end = time.process_time()

                        duration = time_end - time_start
                        difference = theoretical_value - entropy

                        entropy_samples[sample_position].append(entropy)
                        duration_samples[sample_position].append(duration)
                        difference_samples[sample_position].append(difference)

                        #print(f"samples={size_sample}, duration={time_end-time_start}, alpha={alpha}, tested_estimator={entropy}, theoretical_calculation={theoretical_value}, difference={difference}")

                    print(f"{alpha} {size_sample} {sigma} {np.mean(entropy_samples[sample_position])} {np.std(entropy_samples[sample_position])} {np.mean(duration_samples[sample_position])} {np.std(duration_samples[sample_position])} {np.mean(difference_samples[sample_position])} {np.std(difference_samples[sample_position])}", file=fd, flush=True)

def small_test():
    sample_array = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=float)
    input_sample = np.ndarray(shape=sample_array.shape, buffer=sample_array)
    #print(input_sample)
    #print(mutual_inf.renyi_entropy(np.matrix([[1],[2],[3],[4],[5],[6],[7],[8],[9]]), method="LeonenkoProzanto"))
    #print(mutual_inf.renyi_entropy(input_sample, method="Paly"))

    mu = 0
    sigma = 10

    number_samples = 100
    alpha = 0.98
    alphas = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9]
    for alpha in alphas:
        for number_samples in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
            entropy = 0
            data_samples = np.random.normal(mu, sigma, (number_samples, 1))
            time_start = time.process_time()
            #entropy = mutual_inf.renyi_entropy(samples, method="Lavicka", indices_to_use=[1])
            time_end = time.process_time()
            #print(number_samples, time_end-time_start, entropy, Renyi_normal_distribution_1D(sigma, alpha))

            time_start = time.process_time()
            entropy = mutual_inf.renyi_entropy(data_samples, method="LeonenkoProzanto", indices_to_use=[1, 2, 3, 4, 5], alpha=alpha)
            theoretical_value = Renyi_normal_distribution_1D(sigma, alpha)
            difference = abs(theoretical_value-entropy)
            time_end = time.process_time()
            print(f"samples={number_samples}, duration={time_end-time_start}, alpha={alpha}, tested_estimator={entropy}, theoretical_calculation={theoretical_value}, difference={difference}")


if __name__ == "__main__":
    #small_test()
    #complete_test_1D(samples=10, sigmas=[10.0, 50, 100], theoretical_value_function=lambda sigma, alpha: Renyi_student_t_distribution_1D(sigma, sigma, alpha), sample_generator=lambda mu, sigma, size_sample: np.random.standard_t(sigma, size=(size_sample, 1)))
    #sigma_skeleton = np.matrix("5 1; 1 10")
    if len(sys.argv) >= 2:
        dimensions = [int(sys.argv[1])]
    else:
        dimensions = [2, 3, 5, 10, 20, 50]

    print(f"Calculation for dimensions {dimensions}")

    for dimension in dimensions:
        complete_test_ND(filename=f"statistics_{dimension}.txt", samples=200, sigmas=[0.1, 1, 10, 100], sigma_skeleton=np.identity(dimension), sizes_of_sample=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                         theoretical_value_function=lambda sigma, alpha: Renyi_normal_distribution_ND(sigma, alpha),
                         sample_generator=lambda mu, sigma, size_sample: sample_normal_distribution(sigma, size_sample))
