import os
import numpy as np
from sklearn.utils import shuffle


class VADData:
	
	def __init__(self, config, pars, options, scale=True):
		
		folders = config['folders']
		self.data_folder = folders['data_folder']
		self.label_folder = folders['label_folder']
		self.data_use = config['general']['data_size']
		
		model_type = config['general']['model_type']
		self.reshape = (model_type != 'FNN')
		self.data_pars = pars[model_type]['data']
		self.n_steps = self.data_pars['n_steps']
		self.options = options
		
		self.scale = scale
		self.train_start = ['NIS', 'VIT']
		self.dev_start = 'EDI'
		self.test_start = 'CMU'
		
	def load_train_dev(self, shuffle_data=False):
		
		# Get the file names from audio folder:
		data_files = os.listdir(self.data_folder)
		train_files = [file for file in data_files if file[:3] in self.train_start]
		dev_files = [file for file in data_files if file.startswith(self.dev_start)]
		
		X_train, y_train = self.load_from_files(train_files)
		X_dev, y_dev = self.load_from_files(dev_files)
		
		if shuffle_data:
			X_train, y_train = shuffle(X_train, y_train)
			X_dev, y_dev = shuffle(X_dev, y_dev)
			
		# Keep only certain percentage of data:
		n_train = int(self.data_use * X_train.shape[0])
		X_train, y_train = X_train[:n_train], y_train[:n_train]
		n_dev = int(self.data_use * X_dev.shape[0])
		X_dev, y_dev = X_dev[:n_dev], y_dev[:n_dev]
		
		return X_train, y_train, X_dev, y_dev
	
	def load_test(self):
		
		data_files = os.listdir(self.data_folder)
		test_files = [file for file in data_files if file[:3] in self.test_start]
		
		X_test, y_test = self.load_from_files(test_files)
		
		return X_test, y_test
		
	def load_from_files(self, files):
		
		file_data = []
		file_labels = []
		
		for file in files:
			if self.options.verbose:
				print(f'Extracting data from {file}')
			x = np.load(f'{self.data_folder}{file}').astype(np.float32)
			if self.scale:
				x = x - np.mean(x, axis=0) / np.std(x, axis=0)
			y = np.load(f'{self.label_folder}{file}').astype(np.int)
			if self.n_steps:
				x, y = self.add_steps(x, y)
			if self.reshape:
				x = x.reshape((x.shape[0], (self.n_steps*2)+1, -1))
			file_data.append(x)
			file_labels.append(y)
		
		return np.vstack(file_data), np.vstack(file_labels)
	
	def add_steps(self, data, labels):
		# Get n previous features
		backward = self.shift_n_steps(data, 1, self.n_steps+1, keep_original=False)
		# Get n following features
		forward = self.shift_n_steps(data, 1, self.n_steps+1, reverse=True, keep_original=False)
		# Stack them together
		stacked = np.hstack([backward, data, forward])
		# Remove rows with nan values (begining and end)
		nan_mask = np.any(np.isnan(stacked), axis=1)
		stacked = stacked[~nan_mask]
		labels = labels[~nan_mask]
		return stacked, labels
	
	def shift_n_steps(self, arr, start, steps, reverse=False, keep_original=True, fill_value=np.nan):
		initial_arr = arr.copy()
		rng = np.arange(start, steps)
		if reverse:
			rng = -rng
		for i, shift in enumerate(rng):
			if i == 0 and not keep_original:
				arr = self.shift_step(initial_arr, shift, fill_value=fill_value)
			else:
				add_arr = self.shift_step(initial_arr, shift, fill_value=fill_value)
				if reverse:
					arr = np.hstack([arr, add_arr])
				else:
					arr = np.hstack([add_arr, arr])
		return arr
	
	@staticmethod
	def shift_step(arr, num, fill_value=np.nan):
		result = np.empty_like(arr)
		if num > 0:
			result[:num] = fill_value
			result[num:] = arr[:-num]
		elif num < 0:
			result[num:] = fill_value
			result[:num] = arr[-num:]
		else:
			result[:] = arr
		return result