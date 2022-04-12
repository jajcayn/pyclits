#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import sys
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def figures3d_TE(dataset, selector, title, xlabel, ylabel, zlabel, filename, suffix, view=(50, -20), dpi=300):
	fig = plt.figure(figsize=(13, 8))
	ax = fig.add_subplot(1, 1, 1, projection='3d')

	colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]
	markers = ['b', '^']

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)

	row_size = len(dataset['epsilon'].unique())
	xs = dataset[['alpha']]
	ys = dataset[['epsilon']]
	zs = dataset[[selector]]

	try:
		ax.plot_wireframe(np.reshape(xs.values, (-1, row_size)), np.reshape(ys.values, (-1, row_size)), np.reshape(zs.values, (-1, row_size)),
		                  rstride=1, cstride=1, color=colors[0], linewidth=1)
	except Exception as exc:
		print(f"{exc}: Problem D=")

	# Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.legend(loc=1)
	ax.view_init(view[0], view[1])

	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	# plt.draw()
	# plt.show()
	plt.close()
	del fig


def figures3d_surface_TE(dataset, selector, title, xlabel, ylabel, zlabel, filename, suffix, cmap="magma", view=(50, -20), dpi=300):
	fig = plt.figure(figsize=(13, 8))
	ax = fig.add_subplot(1, 1, 1, projection='3d')

	colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]
	markers = ['b', '^']

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)

	row_size = len(dataset['epsilon'].unique())
	xs = dataset[['alpha']]
	ys = dataset[['epsilon']]
	zs = dataset[[selector]]

	try:
		ax.plot_surface(np.reshape(xs.values, (-1, row_size)),
		                np.reshape(ys.values, (-1, row_size)),
		                np.reshape(zs.values, (-1, row_size)),
		                rstride=1,
		                cstride=1,
		                cmap=cmap,
		                linewidth=0,
		                antialiased=False)
	except Exception as exc:
		print(f"{exc}: Problem D=")

	# Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.legend(loc=1)
	ax.view_init(view[0], view[1])

	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	# plt.draw()
	# plt.show()
	plt.close()
	del fig


def minimal_difference(target):
	epsilon_differences = []
	for item in range(0, len(target) - 1):
		epsilon_differences.append(round(target[item + 1] - target[item], 4))
	return min(epsilon_differences)


def figures2d_imshow(dataset, selector, title, xlabel, ylabel, filename, suffix, cmap="magma", dpi=300):
	color_map = matplotlib.cm.get_cmap(cmap)

	fig, ax = plt.subplots(1, 1, figsize=(13, 8))

	ax.set(title=title)
	ax.set(xlabel=xlabel)
	ax.set(ylabel=ylabel)
	ax.grid(True)

	epsilons = dataset['epsilon'].unique()
	alphas = dataset['alpha'].unique()
	epsilon = dataset[['epsilon']]
	alpha = dataset[['alpha']]
	data = dataset[[('epsilon', "", "", "", ""), ('alpha', "", "", "", ""), selector]]
	xs = dataset[['epsilon']].values.reshape((len(alphas), len(epsilons)))
	ys = dataset[['alpha']].values.reshape((len(alphas), len(epsilons)))
	zs = dataset[[selector]].values.reshape((len(alphas), len(epsilons)))
	coords = np.array(list(zip(xs.flatten(), ys.flatten())))

	minimal_epsilon_difference = minimal_difference(epsilons)
	changed_epsilons = np.arange(epsilons[0], epsilons[-1], minimal_epsilon_difference)
	flatten_zs = zs.flatten()

	grid = np.dstack(np.meshgrid(changed_epsilons, alphas)).reshape(-1, 2)
	sampled_data = griddata(coords, flatten_zs, grid, method='nearest')

	number_epsilons = len(epsilons)
	number_alphas = len(alphas)
	epsilon_margin = (max(epsilons) - min(epsilons)) / (number_epsilons * 2.0)
	alpha_margin = (max(alphas) - min(alphas)) / (number_alphas * 2.0)
	extent = [min(epsilons) - epsilon_margin, max(epsilons) + epsilon_margin, min(alphas) - alpha_margin, max(alphas) + alpha_margin]
	ims = ax.imshow(sampled_data.reshape((len(alphas), len(changed_epsilons))), origin="lower", interpolation='nearest', extent=extent, cmap=color_map,
	                aspect='auto')

	fig.colorbar(ims)
	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	# plt.show()
	plt.close()
	del fig


def figures2d_TE_alpha(dataset, selector, title, xlabel, ylabel, filename, suffix, cmap="rainbow", dpi=300, fontsize=15):
	matplotlib.style.use("seaborn")
	fig = plt.figure(figsize=(13, 8))
	ax = fig.add_subplot(1, 1, 1)

	color_map = matplotlib.cm.get_cmap(cmap)

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	codes = dataset['epsilon'].unique()
	list_selector = list(selector)
	list_selector[4] = not selector[4]
	selector_not = tuple(list_selector)
	columns = list(dataset.columns.values)
	if selector_not in columns:
		number_of_datasets = float(2 * len(codes)) - 1
	else:
		number_of_datasets = float(len(codes)) - 1

	order_of_dataset = 0
	for code in codes:
		subselection = dataset.loc[dataset["epsilon"] == code]

		for swap in [True, False]:
			list_selector = list(selector)
			list_selector[4] = swap
			selector = tuple(list_selector)
			if selector in columns:

				label = code.replace("_", "-")
				if swap:
					label_split = label.split("-")
					label = label_split[1] + "-" + label_split[0]

				ys = subselection[['alpha']]
				zs = subselection[[selector]]

				try:
					map_position = order_of_dataset / number_of_datasets
					color = color_map(map_position)
					ax.plot(ys.values, zs.values, linewidth=3, label=label, color=color)
				except Exception as exc:
					print(f"{exc}: Problem D=")

				order_of_dataset += 1

	plt.legend(loc=0, ncol=2, fontsize=fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	plt.close()
	del fig


def figures2d_TE_alpha_errorbar(dataset, selector, error_selector, title, xlabel, ylabel, filename, suffix, view=(70, 120), cmap="rainbow", dpi=300,
                                fontsize=15):
	matplotlib.style.use("seaborn")

	color_map = matplotlib.cm.get_cmap(cmap)

	fig = plt.figure(figsize=(13, 8))
	ax = fig.add_subplot(1, 1, 1)

	markers = ['b', '^']

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	codes = dataset['epsilon'].unique()
	list_selector = list(selector)
	list_selector[4] = not selector[4]
	selector_not = tuple(list_selector)
	columns = list(dataset.columns.values)
	if selector_not in columns:
		number_of_datasets = float(2 * len(codes)) - 1
	else:
		number_of_datasets = float(len(codes)) - 1

	order_of_dataset = 0
	for code in codes:
		subselection = dataset.loc[dataset["epsilon"] == code]

		for swap in [True, False]:
			list_selector = list(selector)
			list_selector[4] = swap
			selector = tuple(list_selector)
			if selector in columns:

				ys = subselection[['alpha']]
				zs = subselection[[selector]]
				error_bar = subselection[[error_selector]].copy()

				error_selector_negative_std = list(error_selector)
				error_selector_negative_std[1] = "-std"
				# error_bar[tuple(error_selector_negative_std)] = error_bar.apply(lambda x: -x, axis=1, raw=True)
				errors = error_bar.copy().T.to_numpy()

				try:
					map_position = order_of_dataset / number_of_datasets
					color = color_map(map_position)
					lims = np.array([True] * ys.size, dtype=bool)
					ax.errorbar(ys.values.flatten(), zs.values.flatten(), yerr=errors.flatten(), linewidth=3, label=code.replace("_", "-"), color=color,
					            ls='dotted')
				except Exception as exc:
					print(f"{exc}: {errors.shape}")
				order_of_dataset += 1

	plt.legend(loc=0, ncol=2, fontsize=fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	plt.close()
	del fig


def figures2d_TE(dataset, selector, title, xlabel, ylabel, filename, suffix, cmap="rainbow", dpi=300, fontsize=15):
	matplotlib.style.use("seaborn")

	color_map = matplotlib.cm.get_cmap(cmap)

	fig = plt.figure(figsize=(13, 8))
	ax = fig.add_subplot(1, 1, 1)

	markers = ['b', '^']

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	alphas = dataset['alpha'].unique()
	mean = int(len(alphas) / 2)
	neghborhood = 5
	# subselected_alphas = alphas[mean - neghborhood:  mean + neghborhood]
	subselected_alphas = [alpha for number, alpha in enumerate(alphas) if (0.70 <= alpha <= 2 and number % 2 == 0)]

	for alpha in subselected_alphas:
		subselection = dataset.loc[dataset["alpha"] == alpha]
		ys = subselection[['epsilon']]
		zs = subselection[[selector]]

		trasform = lambda alpha: (alpha - min(subselected_alphas)) / (max(subselected_alphas) - min(subselected_alphas))
		color = color_map(trasform(alpha))
		row_size = 100
		try:
			ax.plot(ys.values, zs.values, color=color, linewidth=3, label=r'$\alpha={}$'.format(round(alpha, 3)))
		except Exception as exc:
			print(f"{exc}: Problem D=")

	plt.legend(loc=0, ncol=3, fontsize=fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	plt.close()
	del fig


def figures2d_TE_errorbar(dataset, selector, error_selector, title, xlabel, ylabel, filename, suffix, view=(70, 120), cmap="rainbow", dpi=300, fontsize=15):
	matplotlib.style.use("seaborn")

	color_map = matplotlib.cm.get_cmap(cmap)

	fig = plt.figure(figsize=(13, 8))
	ax = fig.add_subplot(1, 1, 1)

	markers = ['b', '^']

	ax.set_title(title)
	ax.set_xlabel(xlabel, fontsize=fontsize)
	ax.set_ylabel(ylabel, fontsize=fontsize)
	# ax.set_yticks([1, 2, 3, 4, 5], ["10", "100", "1000", "10000", "100000"])
	# plt.yticks((1.0, 2.0, 3.0, 4.0, 5.0), ("10", "100", "1000", "10000", "100000"))

	alphas = dataset['alpha'].unique()
	mean = int(len(alphas) / 2)
	neghborhood = 5
	# subselected_alphas = alphas[mean - neghborhood:  mean + neghborhood]
	subselected_alphas = [alpha for number, alpha in enumerate(alphas) if (0.70 <= alpha <= 2 and number % 2 == 0)]

	for alpha in subselected_alphas:
		subselection = dataset.loc[dataset["alpha"] == alpha]
		ys = subselection[['epsilon']]
		zs = subselection[[selector]]
		error_bar = subselection[[error_selector]].copy()

		error_selector_negative_std = list(error_selector)
		error_selector_negative_std[1] = "-std"
		# error_bar[tuple(error_selector_negative_std)] = error_bar.apply(lambda x: -x, axis=1, raw=True)
		errors = error_bar.copy().T.to_numpy()

		trasform = lambda alpha: (alpha - min(subselected_alphas)) / (max(subselected_alphas) - min(subselected_alphas))
		color = color_map(trasform(alpha))
		row_size = 100
		try:
			label = r"${\tiny" + rf"\alpha={round(alpha, 3)}" + r"}$"
			ax.errorbar(ys.values.flatten(), zs.values.flatten(), yerr=errors.flatten(), color=color, linewidth=3,
			            label=label)
		except Exception as exc:
			print(f"{exc}: {errors.shape}")

	# Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.legend(loc=0, ncol=3, fontsize=fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	plt.close()
	del fig


def figures2d_samples_TE(dataset, selector, title, ylabel, filename, suffix, cmap="rainbow", dpi=300):
	matplotlib.style.use("seaborn")

	color_map = matplotlib.cm.get_cmap(cmap)
	alphas = dataset['alpha'].unique()
	epsilons = dataset['epsilon'].unique()
	subselection = dataset.loc[dataset["alpha"] == alphas[0]]
	subselection = subselection.loc[subselection["epsilon"] == epsilons[0]]

	one_subselection = subselection[[selector]]
	number_of_samples = len(subselection[[selector]].values[0, 0])
	mean = int(len(alphas) / 2)
	neghborhood = 5
	subselected_alphas = alphas[mean - neghborhood:  mean + neghborhood]

	for sample in range(number_of_samples):
		fig = plt.figure(figsize=(13, 8))
		ax = fig.add_subplot(1, 1, 1)

		markers = ['b', '^']

		ax.set_title(title)
		ax.set_xlabel(r"$\varepsilon$")
		ax.set_ylabel(ylabel)

		for alpha in subselected_alphas:
			subselection = dataset.loc[dataset["alpha"] == alpha]
			subselection.sort_values(by=['epsilon'], inplace=True)
			# print(subselection)
			ys = subselection[['epsilon']]
			zs = subselection[[selector]]

			trasform = lambda alpha: (alpha - min(subselected_alphas)) / (max(subselected_alphas) - min(subselected_alphas))
			color = color_map(trasform(alpha))
			row_size = 100
			try:
				ax.plot(ys.values, [float(item[0][sample]) for item in zs.values], color=color, linewidth=3, label=r'$\alpha={}$'.format(round(alpha, 3)))
			except Exception as exc:
				print(f"{exc}: Problem D=")

		# Add a color bar which maps values to colors.
		# fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.legend(loc=4)

		plt.savefig(filename.format(sample) + "." + suffix, dpi=dpi, bbox_inches="tight")
		# plt.draw()
		# plt.show()
		plt.close()


def escort_distribution(datasets, columns, title, xlabel, ylabel, filename, suffix, cmap="rainbow", dpi=300):
	matplotlib.style.use("seaborn")

	color_map = matplotlib.cm.get_cmap(cmap)

	fig = plt.figure(figsize=(13, 8))
	markers = ['b', '^']
	fig.suptitle(title)

	for index_dataset, dataset in enumerate(datasets):
		ax = fig.add_subplot(1, len(datasets), index_dataset + 1)
		if index_dataset == 0:
			ax.set_ylabel(ylabel)
		ax.set_xlabel(xlabel)

		for index_column, column in enumerate(columns):
			subselection_x = dataset[["x"]]
			subselection_y = dataset[[str(column)]]

			try:
				color = color_map(index_column / (len(columns) - 1))
				ax.set_yscale("log")
				if index_dataset == 1:
					ax.set_xlim(-0.45, 0.2)
					ax.set_ylim(0.0000000000001, 0.08)
				else:
					ax.set_xlim(-15, 15)
					ax.set_ylim(0.00000001, 2)
				ax.plot(subselection_x.values, subselection_y.values, color=color, linewidth=2, label=r'$\alpha={}$'.format(round(column, 3)))
			except Exception as exc:
				print(f"{exc}: Problem D=")
		if index_dataset == 0:
			ax.legend(loc=1)

	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	plt.close()


def granger_function_plot(dataset, title, xlabel, ylabel, zlabel, filename, suffix, cmap="rainbow", view=(50, -20), dpi=300):
	fig = plt.figure(figsize=(13, 8))
	ax = fig.add_subplot(1, 1, 1, projection='3d')

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	ax.set_zlim(-5, 0)
	ax.set_ylim(1, 10)
	# ax.set_zscale("log")

	row_size = len(dataset['k'].unique())
	xs = dataset['alpha']
	ys = dataset['k']
	zs = dataset['granger']
	Xs = np.reshape(xs.values, (row_size, -1))
	Ys = np.reshape(ys.values, (row_size, -1))
	Zs = np.reshape(zs.values, (row_size, -1))

	try:
		surf = ax.plot_surface(
			Xs,
			Ys,
			Zs,
			rstride=1,
			cstride=1,
			cmap=cmap,
			linewidth=0,
			antialiased=False)
		fig.colorbar(surf, shrink=0.5, aspect=10)
	except Exception as exc:
		print(f"{exc}: Problem D=")

	# plt.legend(loc=1)
	ax.view_init(view[0], view[1])

	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	plt.close()

	del fig


def lyapunov_exponent_plot(dataset, title, xlabel, ylabels, labels, filename, suffix, cmap="rainbow", dpi=300, fontsize=17):
	matplotlib.style.use("seaborn")

	color_map = matplotlib.cm.get_cmap(cmap)
	fig = plt.figure(figsize=(13, 8))

	x = dataset[[0]].values.flatten().tolist()
	ys = [dataset[[i]].values.flatten().tolist() for i in [1, 5, 4]]
	y0 = [0.0] * len(x)
	ys.insert(1, y0)
	# fig.set_title(title)
	ax = fig.add_subplot(1, 1, 1)

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabels)
	ax.set_xlim(0, 0.25)
	ax.set_ylim(-0.13, 0.135)

	for index, y in enumerate(ys):
		color = color_map(index / (len(ys) - 1))
		ax.plot(x, y, linewidth=3, color=color, label=labels[index])

	# plt.legend(loc=3)
	ax.set_xticklabels([0.0, 0.05, 0.1, 0.15, 0.2, 0.25], fontsize=fontsize)
	ax.set_yticklabels([-0.1, -0.05, 0, 0.05, 0.15, 0.2], fontsize=fontsize)
	plt.savefig(filename + "." + suffix, dpi=dpi, bbox_inches="tight")
	plt.close()
	del fig


def process_datasets(processed_datasets, result_dataset, result_raw_dataset, new_columns_base_name="transfer_entropy", take_k_th_nearest_neighbor=5,
                     converter_epsilon=lambda x: float(x)):
	# taking only some nn data to assure that it converge in theory
	files = glob.glob(processed_datasets)
	print(files)
	frames = []
	frames_raw = []
	for file in files:
		epsilon = converter_epsilon(file.split("-")[1].split(".b")[0])
		path = Path(file)

		table = pd.read_pickle(path)
		frame = pd.DataFrame(table)
		frame["epsilon"] = epsilon
		old_columns = frame.columns

		for item in old_columns[:-1]:
			reversed_order = item[4]
			mean_column_name = f"{new_columns_base_name}_{item[1]}_{item[2]}"
			std_column_name = f"{new_columns_base_name}_{item[1]}_{item[2]}"

			if isinstance(item[3], bool):
				bool_column = 3
			else:
				bool_column = 4

			# add mean of entropy

			calculation = frame.apply(lambda row: np.mean(row[item][take_k_th_nearest_neighbor:]), axis=1, raw=False)
			# calculation = frame.apply(lambda row: if any(row) print(row), axis=1, raw=True)
			# calculation = frame.apply(lambda row: float(np.mean(row[item])), axis=1, raw=True)
			if bool_column == 3:
				frame[mean_column_name, "mean", "", item[bool_column], reversed_order] = calculation
			else:
				frame[mean_column_name, "mean", "", "", item[bool_column], reversed_order] = calculation

			# add std of entropy
			calculation = frame.apply(lambda row: float(np.std(row[item][take_k_th_nearest_neighbor:])), axis=1, raw=False)
			if bool_column == 3:
				frame[std_column_name, "std", "", item[bool_column], reversed_order] = calculation
			else:
				frame[mean_column_name, "std", "", "", item[bool_column], reversed_order] = calculation

		# effective transfer entropy
		column_to_use = [item for item in frame.columns.tolist() if
		                 item[bool_column] is False and not ("entropy" in str(item[0]) or "information" in str(item[0]))]
		for item in column_to_use:
			mean_column_name = f"effective_{new_columns_base_name}_{item[1]}_{item[2]}"
			std_column_name = f"effective_{new_columns_base_name}_{item[1]}_{item[2]}"

			if bool_column == 3:
				frame[mean_column_name, "mean", "", False, item[4]] = frame.apply(
					lambda row: float(
						np.mean(
							np.array(row[item][take_k_th_nearest_neighbor:])
							- np.array(row[item[0], item[1], item[2], not item[3], item[4]][take_k_th_nearest_neighbor:])
						)),
					axis=1,
					raw=False)
				frame[std_column_name, "std", "", False, item[4]] = frame.apply(
					lambda row: float(
						np.std(
							np.array(row[item][take_k_th_nearest_neighbor:])
							- np.array(row[item[0], item[1], item[2], not item[3], item[4]][take_k_th_nearest_neighbor:])
						)),
					axis=1,
					raw=False)
				# lambda row: float(
				#        np.std(row[item][take_k_th_nearest_neighbor:])
				#        + np.std(row[item[0], item[1], item[2], not item[3], item[4]][take_k_th_nearest_neighbor:]))
			else:
				frame[mean_column_name, "mean", "", False, item[4]] = frame.apply(
					lambda row: float(
						np.mean(
							np.array(row[item][take_k_th_nearest_neighbor:])
							- np.array(row[item[0], item[1], item[2], item[3], not item[4]][take_k_th_nearest_neighbor:])
						)),
					axis=1,
					raw=False)
				frame[std_column_name, "std", "", False, item[4]] = frame.apply(
					lambda row: float(
						np.std(
							np.array(row[item][take_k_th_nearest_neighbor:])
							- np.array(row[item[0], item[1], item[2], item[3], not item[4]][take_k_th_nearest_neighbor:])
						)),
					axis=1,
					raw=False)

		# balance of entropy
		balance_names = [item for item in frame.columns.tolist() if not bool(item[4]) and "information" not in str(item[0]) and "epsilon" not in str(item[0])]
		for item in balance_names:
			mean_column_name = f"balance_{new_columns_base_name}_{item[1]}_{item[2]}"
			std_column_name = f"balance_{new_columns_base_name}_{item[1]}_{item[2]}"

			frame[mean_column_name, "mean", "", item[3], False] = frame.apply(
				lambda row: float(
					np.mean(
						np.array(row[item][take_k_th_nearest_neighbor:])
						- np.array(row[item[0], item[1], item[2], item[3], not item[4]][take_k_th_nearest_neighbor:])
					)),
				axis=1,
				raw=False)
			frame[std_column_name, "std", "", item[3], False] = frame.apply(
				lambda row: float(
					np.std(
						np.array(row[item][take_k_th_nearest_neighbor:])
						- np.array(row[item[0], item[1], item[2], item[3], not item[4]][take_k_th_nearest_neighbor:])
					)),
				axis=1,
				raw=False)

		# balance of effective entropy
		balance_names = [item for item in frame.columns.tolist() if
		                 not bool(item[4]) and not bool(item[3]) and "information" not in str(item[0]) and "epsilon" not in str(item[0])]
		for item in balance_names:
			mean_column_name = f"balance_effective_{new_columns_base_name}_{item[1]}_{item[2]}"
			std_column_name = f"balance_effective_{new_columns_base_name}_{item[1]}_{item[2]}"

			frame[mean_column_name, "mean", "", item[3], False] = frame.apply(
				lambda row: float(
					np.mean(
						np.array(row[item][take_k_th_nearest_neighbor:])
						- np.array(row[item[0], item[1], item[2], not item[3], item[4]][take_k_th_nearest_neighbor:])
						- np.array(row[item[0], item[1], item[2], item[3], not item[4]][take_k_th_nearest_neighbor:])
						+ np.array(row[item[0], item[1], item[2], not item[3], not item[4]][take_k_th_nearest_neighbor:])
					)),
				axis=1,
				raw=False)
			frame[std_column_name, "std", "", item[3], False] = frame.apply(
				lambda row: float(
					np.std(
						np.array(row[item][take_k_th_nearest_neighbor:])
						- np.array(row[item[0], item[1], item[2], not item[3], item[4]][take_k_th_nearest_neighbor:])
						- np.array(row[item[0], item[1], item[2], item[3], not item[4]][take_k_th_nearest_neighbor:])
						+ np.array(row[item[0], item[1], item[2], not item[3], not item[4]][take_k_th_nearest_neighbor:])
					)),
				axis=1,
				raw=False)

		# dropping the index
		frame = frame.reset_index()

		# print(frame.columns.tolist())
		column = [("alpha", "", "") if "index" == item[0] else item for item in frame.columns.tolist()]
		new_columns = pd.MultiIndex.from_tuples([("alpha", "", "", "", "") if "index" == item[0] else item for item in frame.columns])
		frame.columns = new_columns

		# give names to the columns
		# new_columns = [f"{new_columns_base_name}_{item[1]}_{item[2]}" for item in old_columns[:-1]]
		# column_names = ["alpha"]
		# column_names.extend(new_columns)
		# column_names.append("epsilon")
		# .append(["epsilon"])
		# frame.columns = column_names

		# selection of columns
		columns = [item for item in frame.columns.tolist() if
		           "mean" in str(item[1]) or "std" in str(item[1]) or "alpha" in str(item[0]) or "epsilon" in str(item[0])]
		frame_with_processed_results = frame[columns]

		columns = [item for item in frame.columns.tolist() if
		           isinstance(item[0], float) or "alpha" in str(item[0]) or "epsilon" in str(item[0])]
		frame_with_raw_results = frame[columns]
		# print(frame)
		# if item[0] not in ["alpha", "epsilon"] else item[0:3]
		columns = [str(item[1]) + "_" + str(item[2]) + "_" + str(item[3]) if isinstance(item[0], float) else item[0] for item in
		           frame_with_raw_results.columns.tolist()]
		frame_with_raw_results.columns = columns

		# append frame for processing
		frames.append(frame_with_processed_results)
		frames_raw.append(frame_with_raw_results)

	# join the table
	join_table = pd.concat(frames, ignore_index=True)
	try:
		join_table_raw = pd.concat(frames_raw, ignore_index=True)
	except:
		print("Problem with columns")
		first_frame = frames_raw[0]
		for file, frame in zip(files, frames_raw):
			comparison = len(frame.columns.tolist()) == len(first_frame.columns.tolist())
			if not comparison:
				print(file, frame.columns, comparison)
		sys.exit(1)

	# print(join_table)
	index_alpha = join_table.columns.tolist()
	pivot_table = pd.pivot_table(join_table, index=[index_alpha[0], index_alpha[1]])
	print(pivot_table, join_table.columns.tolist())

	print(join_table_raw)
	index_alpha = join_table_raw.columns.tolist()
	pivot_table_raw = join_table_raw.set_index([index_alpha[0], index_alpha[-1]])
	# pd.pivot_table(join_table_raw, index=[index_alpha[0], index_alpha[1]])
	print(pivot_table_raw)

	# print(pivot_table[["transfer_entropy_15_5_mean"]])
	TE = pivot_table.reset_index()
	TE_raw = pivot_table_raw.reset_index()

	TE.to_pickle(result_dataset)
	TE_raw.to_pickle(result_raw_dataset)

	return TE, [item for item in join_table.columns.tolist() if "mean" in str(item[1])], TE_raw
