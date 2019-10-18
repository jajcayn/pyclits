
import numpy as np
import time
import mutual_inf
import math

time_start = time.process_time()

def Renyi_normal_distribution(sigma, alpha):
    return math.log2(2*math.pi) / 2 + math.log2(sigma) + math.log2(alpha) / (1 - alpha) / 2

if __name__ == "__main__":
    import time

    sample_array = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=float)
    input_sample = np.ndarray(shape=sample_array.shape, buffer=sample_array)
    #print(input_sample)
    print(mutual_inf.renyi_entropy(np.matrix([[1],[2],[3],[4],[5],[6],[7],[8],[9]]), method="Lavicka"))
    print(mutual_inf.renyi_entropy(input_sample, method="Lavicka"))

    mu = 0
    sigma = 10

    number_samples = 100
    alpha = 0.98
    for number_samples in [10, 20, 50, 100, 200, 500, 1000, 2000]:
        entropy = 0
        samples = np.random.normal(mu, sigma, (number_samples, 1))
        time_start = time.process_time()
        #entropy = mutual_inf.renyi_entropy(samples, method="Lavicka", indices_to_use=[1])
        time_end = time.process_time()
        print(number_samples, time_end-time_start, entropy, Renyi_normal_distribution(sigma, alpha))

        time_start = time.process_time()
        entropy = mutual_inf.renyi_entropy(samples, method="Paly", indices_to_use=[1], alpha=alpha)
        time_end = time.process_time()
        print(number_samples, time_end-time_start, entropy, Renyi_normal_distribution(sigma, alpha))
