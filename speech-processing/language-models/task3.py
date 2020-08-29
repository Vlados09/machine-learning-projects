import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Smoothing factor for numerical stability
EPS = np.finfo(float).eps


def interpolate_lm(dev_lm, eval_lm, thresh=0.001, print_progress=True):
	"""
	A method for interpolating between language models
	:param dev_lm: language models for weight estimation (n_grams * n_lm)
	:param eval_lm: language models for evaluation (n_grams * n_lm)
	:param thresh: threshold to stop learning at
	:param print_progress: where as to print out the output
	:return: learnt weights and histories for the training
	"""
	
	# Initialise the weights
	weights = init_weights(dev_lm, init='ppl')
	if print_progress:
		print(f'Initial weights: \n {weights} \n')
	
	dev_ppl_hist = []
	eval_ppl_hist = []
	iteration = 0
	ppl_diff = thresh + EPS
	
	while ppl_diff > thresh:
		
		# Compute PPL for tacking:
		dev_ppl = compute_ppl(dev_lm, weights=weights)
		eval_ppl = compute_ppl(eval_lm, weights=weights)
		
		# Update the weights
		weights = update_weights(dev_lm, weights)
		
		if print_progress:
			print(f'Iteration: {iteration}| Development PPL: {dev_ppl:.3f}| Evaluation PPL: {eval_ppl:.3f}')
		
		if iteration:
			assert dev_ppl_hist[-1] >= dev_ppl, 'Convergence guarantee has not been met...'
			ppl_diff = dev_ppl_hist[-1] - dev_ppl
		
		iteration += 1
		dev_ppl_hist.append(dev_ppl)
		eval_ppl_hist.append(eval_ppl)
		
	return weights, dev_ppl_hist, eval_ppl_hist


def init_weights(lm, init='ppl'):
	"""
	Weight initialisation procedure with 3 different approaches
	:param lm: language models for which the weights should be initialised (n_ngrams * n_lm)
	:param init: string determining which initialisation to use
	:return: weights (n_lm * 1)
	"""
	n_lm = lm.shape[1]
	if init == 'uniform':
		weights = np.random.uniform(0, 0.01, size=n_lm)
	elif init == 'ppl':
		weights = np.empty(n_lm)
		for i in range(n_lm):
			# Set weights to inverse perplexity of each LM
			weights[i] = 1/compute_ppl(lm[:, i])
	else:
		weights = np.ones(n_lm)
	# Normalise the weights to sum up to 1:
	weights = weights / np.sum(weights)
	return weights


def update_weights(lm, weights):
	"""
	Update the weights using posterior probability
	:param lm: language models for which wights should be updated (n_ngrams * n_lm)
	:param weights: current weights to be updated
	:return: updated weights
	"""
	weighted_lm = weights * lm
	lm = np.sum(weighted_lm, axis=1) + EPS
	posterior = weighted_lm / lm.reshape(-1, 1)
	weights = (1 / lm.shape[0]) * np.nansum(posterior, axis=0)
	return weights


def compute_ppl(lm, weights=None):
	"""
	Computer PPL for a given language model.
	:param lm: language model for which PPL should be computed
	:param weights: (optional) if weights are provided computes interpolated PPL
	:return: PPL measure of a language model
	"""
	if weights is not None:
		weighted_lm = weights * lm
		lm = np.sum(weighted_lm, axis=1)
	ppl = np.exp((-1 / lm.shape[0]) * np.sum(np.log(lm + EPS)))
	return ppl


def plot_results(dev_ppl_hist, eval_ppl_hist):
	
	plt.figure()
	plt.plot(dev_ppl_hist, label='dev')
	plt.plot(eval_ppl_hist, label='eval')
	plt.xlabel('Iteration')
	plt.ylabel('PPL')
	plt.legend()
	plt.show()
	
	
def get_data(folder_path):
	"""
	Extract language models from individual files and put them together in a NumPy array
	:param folder_path: folder to extract language models from
	:return: language models np.array and list of file names for visualisations
	"""
	language_models = []
	file_names = []
	for file in os.listdir(folder_path):
		language_models.append(pd.read_csv(f'{folder_path}{file}').values)
		file_names.append(file.split('.')[0])
	language_models = np.hstack(language_models)
	return language_models, file_names


def stupid_backoff(lm):
	"""
	Performs a simple backoff procedure
	"""
	for i in range(lm.shape[0]):
		for j in range(lm.shape[1]):
			if not lm[i, j]:
				lm[i, j] = 0.4 * lm[i-1, j]
	
	return lm


def parse_options():
	parser = ArgumentParser(description="This is a command line interface (CLI) for Task 3 of Speech module",
	                        epilog="Vlad Bondarenko, 2020-05-10")
	parser.add_argument("dev_data", action="store", type=str, metavar="<path-to-file>",
	                    help="Location of development language models")
	parser.add_argument("eval_data", action="store", type=str, metavar="<path-to-file>",
	                    help="Location of evaluation language models")
	options = parser.parse_args()

	if not (os.path.exists(options.dev_data) & os.path.exists(options.eval_data)):
		raise FileNotFoundError('Provided path do not exist')
	
	return options


if __name__ == '__main__':
	
	opt = parse_options()
	dev_data, dev_names = get_data(opt.dev_data)
	eval_data, eval_names = get_data(opt.eval_data)
	assert dev_names == eval_names, 'Files in evaluation and development set differ'
	
	dev_data = stupid_backoff(dev_data)
	eval_data = stupid_backoff(eval_data)
	
	w, dev_ppl_h, eval_ppl_h = interpolate_lm(dev_data, eval_data, thresh=0.01)
	
	print('\n')
	for weight, name in zip(w, dev_names):
		print(f'Weight for {name.upper()} language model is {weight:.3f}')
	print('\n')
	
	print(f'Final Development PPL: {dev_ppl_h[-1]:.3f}')
	print(f'Final Evaluation PPL: {eval_ppl_h[-1]:.3f}')
	print(f'Difference in two is: {(eval_ppl_h[-1] - dev_ppl_h[-1]):.3f}')
	print(f'Calculated with {dev_data.shape[0]} ngrams')
	
	plot_results(dev_ppl_h, eval_ppl_h)