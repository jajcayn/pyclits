#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import pandas
import scipy.special as special
import pandas as pd
import processing_datasets
import math
import matplotlib.pyplot as plt

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif",
	"font.serif": ["Palatino"],
})


def granger_function(alpha, k, l):
	if alpha == 1:
		return numpy.nan

	xi = (1. / (1. - alpha)) - (k / 2.0)
	dzeta = (1. / (1. - alpha)) - ((k + l) / 2.0)
	if numpy.isinf(dzeta) or numpy.isinf(xi):
		print(alpha, k, l)
	return numpy.log2(
		numpy.power(math.e, special.gammaln(xi - 0.5) - special.gammaln(xi))  # math.gamma(xi - 0.5) / math.gamma(xi)
		* numpy.power(math.e, xi * numpy.log(xi - 1) - (xi - 0.5) * numpy.log(xi - 1.5))  # * numpy.power(xi - 1, xi) / numpy.power(xi - 1.5, xi - 0.5)
		* math.gamma(dzeta) / math.gamma(dzeta - 0.5)
		* numpy.power(math.e, (dzeta - 0.5) * numpy.log(dzeta - 1.5) - dzeta * numpy.log(dzeta - 1))
		# * numpy.power(dzeta - 1.5, dzeta - 0.5) / numpy.power(dzeta - 1, dzeta)
	)


if __name__ == "__main__":
	data = []
	for k in numpy.arange(1, 11, 0.01):
		number_of_added_rows = 0
		n = 500. + 1
		limit_alpha = 0.989
		difference = ((3. + k) / (5. + k) - limit_alpha) / n
		for alpha in numpy.arange(limit_alpha, (3. + k) / (5. + k), difference):
			try:
				granger = granger_function(alpha, k, 2)
				if not numpy.isnan(granger):
					row = [alpha, k, granger]
					number_of_added_rows = number_of_added_rows + 1
					if number_of_added_rows <= int(n):
						data.append(row)
					else:
						print("data dropped")

			except Exception as exc:
				print(f"problem: {k}, {alpha} \n {exc}")
		print(f"added row for k={k} {number_of_added_rows}")

	df = pandas.DataFrame(data, columns=['alpha', 'k', 'granger'])
	processing_datasets.granger_function_plot(
		df,
		"",
		"""\\LARGE $\\alpha$""",
		"""""",  # """\\Large $k$""",
		"""""",  # """\\LARGE $I_\\alpha (Z_\\alpha^{1,1} : Z_\\alpha^{1,l} \\vert Z_\\alpha^{1,k})$""",
		"granger", "png", view=(20, 100)
	)
