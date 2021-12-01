#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import time
import mpmath
import logging
import collections
import numpy as np
import numpy.random as random
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import scipy.special as scipyspecial


def graph_calculation_Paly(data, **kwargs):
    tree_x = KDTree(data, leaf_size=kwargs["leaf_size"], metric=kwargs["metric"])
    distances = tree_x.query(data, k=kwargs["maximal_index"], return_distance=True, dualtree=kwargs["dualtree"])
    selected_distances = distances[0][:, kwargs["indices_to_use"]]
    flatten_distances = selected_distances.flatten()
    power_of_distances = np.power(flatten_distances, kwargs["power_of_distance_data"])
    L_p_V_data = np.sum(power_of_distances)

    return L_p_V_data


def graph_calculation_preparation(data, **kwargs):
    if "leaf_size" in kwargs:
        leaf_size = kwargs["leaf_size"]
    else:
        leaf_size = 40

    if "metric" in kwargs:
        metric = kwargs["metric"]
    else:
        metric = "euclidean"

    print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} * shape of data which will be used to construct tree: {data.shape}", flush=True)
    tree_x = KDTree(data, leaf_size=leaf_size, metric=metric)

    return tree_x


def graph_calculation_within_distance_Lavicka(data, radii, **kwargs):
    tree_x = KDTree(data, leaf_size=kwargs["leaf_size"], metric=kwargs["metric"])
    distances = tree_x.query_radius(data, radii, return_distance=True, count_only=False)

    return distances


def special(k, q, d, N, p0, p1, p, e0, e1):
    value = p0*e1-p1*e0
    return pow(p, 1+k-q) / (1+k-q) * pow((p0-p1)/(p0*e1-p1*e0), d*(1-q)) * mpmath.appellf1(1+k-q, 1+k-N, d * (1-q), 2+k-q, p, p*(e0-e1) / (p1*e0-p0*e1))


def renyi_entropy_Lavicka(dataset_x: np.matrix, alpha=1, leaf_size=15, metric="chebyshev", dualtree=True,
                          sample_size=1000, indices_to_use=[3, 4], **kwargs):
    shape_of_data = dataset_x.shape
    maximal_index = max(indices_to_use) + 1
    length_of_data = shape_of_data[0]
    dimension_of_data = shape_of_data[1]

    kdtree = graph_calculation_preparation(dataset_x, **locals())
    entropy = 0

    distances = kdtree.query(dataset_x, k=kwargs["maximal_index"], return_distance=True, dualtree=dualtree, breadth_first=True)

    for index_of_distances, use_index in enumerate(indices_to_use):
        selected_distances = distances[:, index_of_distances]

        # calculation of PDF
        counter = collections.Counter(selected_distances)
        ordered_distances = sorted(list(counter.keys()))

        divisor = float(len(selected_distances))

        # integration over PDF
        previous_probability = 0
        # save value to prevent problems at start
        previous_distance = 0
        for distance in ordered_distances:
            actual_distance = distance
            actual_probability = previous_probability + float(counter[distance]) / divisor

            addition_to_entropy = (special(use_index, alpha, dimension_of_data, divisor, previous_probability, actual_probability, actual_probability, previous_distance, actual_distance)
                - special(use_index, alpha, dimension_of_data, divisor, previous_probability, actual_probability, previous_probability, previous_distance, actual_distance))
            entropy += addition_to_entropy

            previous_distance = actual_distance
            previous_probability = actual_probability

    return entropy/len(indices_to_use)


def renyi_entropy_LeonenkoProzanto(dataset_x: np.matrix, **kwargs):
    if "indices_to_use" in kwargs:
        indices_to_use = kwargs["indices_to_use"]
    else:
        indices_to_use = [3, 4]
        kwargs["indices_to_use"] = indices_to_use

    if "alphas" in kwargs:
        alphas = kwargs["alphas"]
    else:
        alphas = [1]

    if "transpose" in kwargs:
        transpose = kwargs["transpose"]
    else:
        transpose = False

    if "dualtree" in kwargs:
        dualtree = kwargs["dualtree"]
    else:
        dualtree = True

    if transpose:
        dataset_x = dataset_x.T

    shape_of_data = dataset_x.shape
    kwargs["maximal_index"] = max(indices_to_use) + 1
    length_of_data = shape_of_data[0]
    kwargs["dimension_of_data"] = shape_of_data[1]

    kdtree = graph_calculation_preparation(dataset_x, **kwargs)

    results = {}

    t0 = time.process_time()
    distances = kdtree.query(dataset_x, k=kwargs["maximal_index"], return_distance=True, dualtree=dualtree, breadth_first=True)
    t1 = time.process_time()
    duration = t1 - t0
    del kdtree

    print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} * * Calculation of distances [s]: {duration}", flush=True)

    t0 = time.process_time()
    for alpha in alphas:
        try:
            if alpha == 1.:
                result = entropy_sum_Shannon_LeonenkoProzanto(dataset_x, distances, **kwargs)
            else:
                entropy_sum = entropy_sum_generic_LeonenkoProzanto(dataset_x, distances, alpha, **kwargs)

                # here we take natural logarithm instead of
                if kwargs["arbitrary_precision"]:
                    result = [kwargs["logarithm"](item) / (1.0 - alpha) for item in entropy_sum]
                else:
                    entropy_sum = entropy_sum.tolist()
                    result = [kwargs["logarithm"](item) / (1.0 - alpha) for item in entropy_sum]

            results[alpha] = result
        except Exception as exc:
            print(f"{exc.args[0]}")

    t1 = time.process_time()
    duration = t1 - t0
    print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} * * Calculation of entropy [s]: {duration}", flush=True)

    return results


def tsallis_entropy_LeonenkoProzanto(dataset_x: np.matrix, alpha=1, **kwargs):
    if alpha == 1:
        return entropy_sum_Shannon_LeonenkoProzanto(dataset_x, alpha, **kwargs)
    else:
        return (1 - entropy_sum_generic_LeonenkoProzanto(dataset_x, alpha, **kwargs)) / (1 - alpha)


def entropy_sum_generic_LeonenkoProzanto(dataset_x: np.matrix, distances, alpha=1, **kwargs):
    indices_to_use = kwargs["indices_to_use"]
    dimension_of_data = kwargs["dimension_of_data"]

    if kwargs["arbitrary_precision"]:
        entropy = [mpmath.mpf("0.0") for index in range(len(indices_to_use))]
    else:
        entropy = np.zeros(len(indices_to_use))

    selected_distances = distances[0][:, kwargs["indices_to_use"]]
    del distances

    for index_of_distances, use_index in enumerate(indices_to_use):
        subselected_distances = selected_distances[:, index_of_distances]

        number_of_data = float(len(dataset_x))

        try:
            if kwargs["arbitrary_precision"]:
                one_minus_alpha = 1.0 - alpha
                exponent = dimension_of_data * one_minus_alpha
                addition_to_entropy = mpmath.mpf('0.0')
                if exponent < 0:
                    # dealing with distance 0
                    subselected_distances = np.array([item for item in subselected_distances if item > 0])

                shape = subselected_distances.shape
                for index in range(shape[0]):
                    addition = mpmath.power(subselected_distances[index], exponent)
                    addition_to_entropy += addition

                multiplicator_gamma = mpmath.gammaprod([use_index], [use_index + one_minus_alpha])
                multiplicator = multiplicator_gamma * mpmath.power(mpmath.pi, dimension_of_data / 2.0 * one_minus_alpha) * mpmath.power(number_of_data - 1,
                                                                                                                                        one_minus_alpha) / number_of_data / mpmath.power(
                    mpmath.gamma(dimension_of_data / 2.0 + 1), one_minus_alpha)

                entropy[index_of_distances] += multiplicator * addition_to_entropy
            else:
                maximum_distance = max(subselected_distances)
                one_minus_alpha = 1.0 - alpha
                exponent = dimension_of_data * one_minus_alpha
                scaled_distances = subselected_distances / maximum_distance
                if exponent < 0:
                    # dealing with distance 0
                    scaled_distances = np.array([item for item in scaled_distances if item > 0])

                max_multiplicator = np.power(maximum_distance, exponent)
                power = np.power(scaled_distances, exponent)
                sum_of_power_of_distances = np.sum(power) * max_multiplicator

                multiplicator_exp_logarithms = np.exp(scipyspecial.gammaln(use_index) - scipyspecial.gammaln(use_index + one_minus_alpha)
                                                      - one_minus_alpha * scipyspecial.gammaln(dimension_of_data / 2.0 + 1.0))
                multiplicator = multiplicator_exp_logarithms * np.power(np.pi, exponent / 2.0) * np.power(number_of_data - 1, one_minus_alpha) / number_of_data

                entropy[index_of_distances] += multiplicator * sum_of_power_of_distances
        except Exception as exc:
            print(f"Exception happened: {exc.exc_info()[0]} {alpha} {use_index}")

    return entropy


def entropy_sum_Shannon_LeonenkoProzanto(dataset_x: np.matrix, distances, **kwargs):
    indices_to_use = kwargs["indices_to_use"]
    dimension_of_data = kwargs["dimension_of_data"]

    if "arbitrary_precision" in kwargs and kwargs["arbitrary_precision"]:
        entropy = [mpmath.mpf("0.0") for index in range(len(indices_to_use))]
    else:
        entropy = np.zeros(len(indices_to_use))

    distances_used_by_index = distances[0][:, kwargs["indices_to_use"]]

    for index_of_distances, use_index in enumerate(indices_to_use):
        subselected_distances = distances_used_by_index[:, index_of_distances]

        number_of_data = float(len(dataset_x))

        if kwargs["arbitrary_precision"]:
            addition_to_entropy = mpmath.mpf('0.0')
            subselected_distances = np.array([item for item in subselected_distances if item > 0])
            shape = subselected_distances.shape
            for index in range(shape[0]):
                addition_to_entropy += kwargs["logarithm"](subselected_distances[index])

            addition_to_entropy *= dimension_of_data / number_of_data

            digamma = mpmath.digamma(use_index)
            argument_log = mpmath.power(mpmath.pi, dimension_of_data / 2.0) / mpmath.gamma(dimension_of_data / 2.0 + 1) * mpmath.exp(-digamma) * (
                        number_of_data - 1)

            entropy[index_of_distances] += addition_to_entropy + kwargs["logarithm"](argument_log)
        else:
            # dealing with distance 0 - log then diverges
            subselected_distances = np.array([item for item in subselected_distances if item > 0])

            addition_to_entropy = np.sum(kwargs["logarithm"](subselected_distances)) * dimension_of_data / number_of_data

            digamma = scipyspecial.digamma(use_index)
            argument_log = np.power(np.pi, dimension_of_data / 2.0) / scipyspecial.gamma(dimension_of_data / 2.0 + 1) * np.exp(-digamma) * (number_of_data - 1)

            entropy[index_of_distances] += addition_to_entropy + kwargs["logarithm"](argument_log)

    if not kwargs["arbitrary_precision"]:
        entropy = entropy.tolist()

    return entropy


def renyi_entropy_Paly(dataset_x: np.matrix, alpha=0.75, leaf_size = 15, metric="chebyshev", dualtree=True, sample_size=1000, indices_to_use=[3,4], **kwargs):
    """
    Calculation of Renyi entropy

    :param dataset_x:
    :return:

    According to D.Pal, B. Poczos, C. Szepesvari, Estimation of Renyi Entropy and Mutual Information Based on Generalized Nearest-Neighbor Graphs, 2010.
    """
    if "alphas" in kwargs:
        alphas = kwargs["alphas"]
    else:
        alphas = [1]

    for alpha in alphas:
        results = {}
        if 0.5 < alpha < 1:
            shape_of_data = dataset_x.shape
            maximal_index = max(indices_to_use) + 1
            length_of_data = shape_of_data[0]
            dimension_of_data = shape_of_data[1]
            power_of_distance_data = dimension_of_data * (1 - alpha)

            L_p_V_data = graph_calculation_Paly(dataset_x, **locals())

            random_sample_of_array = random.uniform(size=(sample_size, dimension_of_data))
            L_p_V_sample = graph_calculation_Paly(random_sample_of_array, **locals())

            gamma = L_p_V_sample / np.power(sample_size, 1 - power_of_distance_data / dimension_of_data)

            entropy = 1 / (1 - alpha) * np.log(
                L_p_V_data / (gamma * np.power(length_of_data, 1 - power_of_distance_data / dimension_of_data)))

            results[alpha] = [entropy]
        else:
            raise Exception("Paly method works for alpha in range (0.5,1)")

    return results


def renyi_entropy(*args, **kwargs):
    # preparation of logarithm
    if "arbitrary_precision" not in kwargs:
        raise BaseException("arbitrary_precision missing in kwargs")
    else:
        if kwargs[ "arbitrary_precision"]:
            if "base_of_logarithm" in kwargs:
                kwargs["logarithm"] = lambda x: mpmath.log(x, kwargs["base_of_logarithm"])
            else:
                kwargs["logarithm"] = lambda x: mpmath.log(x)
        else:
            if "base_of_logarithm" in kwargs:
                kwargs["logarithm"] = lambda x: np.log(x, kwargs["base_of_logarithm"])
            else:
                kwargs["logarithm"] = lambda x: np.log(x)

    if "method" in kwargs:
        if kwargs["method"] == "Paly" or kwargs["method"] == "GeneralizedNearestNeighbor":
            return renyi_entropy_Paly(*args, **kwargs)
        elif kwargs["method"] == "Lavicka" or kwargs["method"] == "NearestNeighbor":
            return renyi_entropy_Lavicka(*args, **kwargs)
        elif kwargs["method"] == "LeonenkoProzanto":
            return renyi_entropy_LeonenkoProzanto(*args, **kwargs)
        else:
            logging.error("Wrong method was choosen.")
            raise Exception("Wrong method was choosen.")
    else:
        logging.error("No method was choosen.")
        raise Exception("No method was choosen.")


def renyi_mutual_entropy(data_x, data_y, **kwargs):
    if "axis_to_join" in kwargs:
        axis_to_join = kwargs["axis_to_join"]
    else:
        axis_to_join = 0

    marginal_entropy_x = renyi_entropy(data_x, **kwargs)
    marginal_entropy_y = renyi_entropy(data_y, **kwargs)
    joint_dataset = np.concatenate((data_x, data_y), axis=axis_to_join)
    entropy_xy = renyi_entropy(joint_dataset, **kwargs)

    results = {}
    for alpha in kwargs["alphas"]:
        result = marginal_entropy_x[alpha] + marginal_entropy_y[alpha] - entropy_xy[alpha]
        results[alpha] = result

    return results


def renyi_transfer_entropy(data_x, data_x_hist, data_y, **kwargs):
    if "enhanced_calculation" in kwargs:
        enhanced_calculation = kwargs["enhanced_calculation"]
    else:
        enhanced_calculation = True

    if "axis_to_join" in kwargs:
        axis_to_join = kwargs["axis_to_join"]
    else:
        axis_to_join = 0

    results = {}
    if enhanced_calculation:
        joint_dataset = np.concatenate((data_x, data_x_hist), axis=axis_to_join)
        entropy_present_X_history_X = renyi_entropy(joint_dataset, **kwargs)

        joint_dataset = np.concatenate((data_x_hist, data_y), axis=axis_to_join)
        entropy_history_X_history_Y = renyi_entropy(joint_dataset, **kwargs)

        joint_dataset = np.concatenate((data_x, data_x_hist, data_y), axis=axis_to_join)
        entropy_joint = renyi_entropy(joint_dataset, **kwargs)

        entropy_history_X = renyi_entropy(data_x_hist, **kwargs)

        for alpha in kwargs["alphas"]:
            result = [
                entropy_present_X_history_X[alpha][index] + entropy_history_X_history_Y[alpha][index] - entropy_joint[alpha][index] - entropy_history_X[alpha][
                    index]
                for index in range(len(kwargs["indices_to_use"]))]
            results[alpha] = result
        return results
    else:
        joint_dataset = np.concatenate((data_x_hist, data_y), axis=axis_to_join)

        joint_part = renyi_mutual_entropy(data_x, joint_dataset, **kwargs)
        marginal_part = renyi_mutual_entropy(data_x, data_x_hist, **kwargs)

        results = {}
        for alpha in kwargs["alphas"]:
            result = [joint_part[alpha][index] - marginal_part[alpha][index] for index in range(len(kwargs["indices_to_use"]))]
            results[alpha] = result

        return results


def renyi_conditional_information_transfer(data_x_fut, data_x_hist, data_y, **kwargs):
    if "enhanced_calculation" in kwargs:
        enhanced_calculation = kwargs["enhanced_calculation"]
    else:
        enhanced_calculation = True

    if "axis_to_join" in kwargs:
        axis_to_join = kwargs["axis_to_join"]
    else:
        axis_to_join = 0

    results = {}
    if enhanced_calculation:
        joint_dataset = np.concatenate((data_x_fut, data_x_hist), axis=axis_to_join)
        entropy_present_X_history_X = renyi_entropy(joint_dataset, **kwargs)

        joint_dataset = np.concatenate((data_x_hist, data_y), axis=axis_to_join)
        entropy_history_X_history_Y = renyi_entropy(joint_dataset, **kwargs)

        joint_dataset = np.concatenate((data_x_fut, data_x_hist, data_y), axis=axis_to_join)
        entropy_joint = renyi_entropy(joint_dataset, **kwargs)

        entropy_history_X = renyi_entropy(data_x_hist, **kwargs)

        for alpha in kwargs["alphas"]:
            try:
                result = [
                    entropy_present_X_history_X[alpha][index] + entropy_history_X_history_Y[alpha][index] - entropy_joint[alpha][index] -
                    entropy_history_X[alpha][
                        index]
                    for index in range(len(kwargs["indices_to_use"]))]
                results[alpha] = result
            except KeyError as exc:
                print(f"Key {alpha} is missing: {exc.with_traceback()}")
        return results
    else:
        joint_dataset = np.concatenate((data_x_hist, data_y), axis=axis_to_join)

        joint_part = renyi_mutual_entropy(data_x_fut, joint_dataset, **kwargs)
        marginal_part = renyi_mutual_entropy(data_x_fut, data_x_hist, **kwargs)

        results = {}
        for alpha in kwargs["alphas"]:
            result = [joint_part[alpha][index] - marginal_part[alpha][index] for index in range(len(kwargs["indices_to_use"]))]
            results[alpha] = result

        return results


def conditional_transfer_entropy(data_x, data_y, data_z, **kwargs):
    joint_dataset_xz = np.concatenate(data_x, data_z, axis=1)
    marginal_entropy_xz = renyi_entropy(joint_dataset_xz, **kwargs)

    marginal_entropy_z = renyi_entropy(data_z, **kwargs)

    joint_dataset_xyz = np.concatenate(data_x, data_x, data_z, axis=1)
    entropy_xyz = renyi_entropy(joint_dataset_xyz, **kwargs)

    joint_dataset_yz = np.concatenate(data_y, data_y, axis=1)
    entropy_xy = renyi_entropy(joint_dataset_yz, **kwargs)

    return marginal_entropy_xz - marginal_entropy_z - entropy_xyz + entropy_xy


if __name__ == "__main__":
    sample_array = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=float)
    input_sample = np.ndarray(shape=sample_array.shape, buffer=sample_array)
    #print(input_sample)
    print(renyi_entropy(np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]]), method="LeonenkoProzanto", arbitrary_precision=False))
    print(renyi_entropy(input_sample, method="LeonenkoProzanto", arbitrary_precision=False))

    mu = 0
    sigma = 10
    number_samples = 100

    samples = np.random.normal(mu, sigma, (number_samples, 1))
    print(renyi_entropy(samples, method="LeonenkoProzanto", maximal_index=20, arbitrary_precision=False))

