import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import seaborn
seaborn.set()
import numpy as np
import tensorflow as tf
from configparser import ConfigParser
from argparse import ArgumentParser
from bayes_opt import BayesianOptimization

from VADModel import VADModel
from VADData import VADData
from VADEval import VADEval


class Task4:
	
	def __init__(self):
		
		# Ensure those parameters always converted to int:
		self.int_pars = ['l1', 'l2', 'batch_size', 'epochs', 'n_steps']
		self.options = self.parse_options()
		self.config = self.load_config()
		self.pars = self.load_pars()
		
		self.folders = self.config['folders']
		self.model_type = self.config['general']['model_type']
	
	def run_training(self, **kwargs):
		"""
		Trains the VAD model
		:param kwargs: model parameters in a form of a dictionary
		:return: negative EER for use during optimisation
		"""
		
		# Ensure same random state for each run
		seed = self.config['general']['seed']
		np.random.seed(seed)
		tf.random.set_seed(seed)
		
		if kwargs:
			self.adjust_pars(kwargs)
		
		# Get training and valiation/development data:
		data = VADData(self.config, self.pars, self.options)
		X_train, y_train, X_dev, y_dev = data.load_train_dev(shuffle_data=True)
		
		# Penalize class imbalances during training:
		class_w = (np.sum(y_train) + np.sum(y_dev))/(y_train.size + y_dev.size)
		self.pars[self.model_type]['train']['class_weight'] = [class_w, 1-class_w]
		
		# Initialise and train the model
		model = VADModel(self.config, self.pars, self.options, X_train.shape)
		model.train(X_train, y_train, X_dev, y_dev)
		
		# Perform evaluation on the development set
		evaluation = VADEval(self.config, self.options)
		dev_pred = model.predict(X_dev)
		eer = evaluation.evaluate(y_dev, dev_pred)
		
		if self.options.save:
			model.save()
		
		return -eer
	
	def run_evaluation(self):
		"""
		Perform evaluation of the saved model
		"""
		data = VADData(self.config, self.pars, self.options)
		X_test, y_test = data.load_test()
		
		# Initialise and load the model
		model = VADModel(self.config, self.pars, self.options, X_test.shape)
		model.load()
		
		# Perform final evaluation on the test set
		evaluation = VADEval(self.config, self.options)
		test_pred = model.predict(X_test)
		eer = evaluation.evaluate(y_test, test_pred)
	
	def optimize(self):
		"""
		Method for running Bayesian optimisation
		:return: optimizer object with results stored for each iteration
		"""
		self.options.opt = False
		pbounds = {}
		for k0, v0 in self.pars.items():
			for k1, v1 in v0.items():
				try:
					for k2, v2 in v1.items():
						if isinstance(v2, (list, tuple)):
							pbounds[k2] = v2
				except AttributeError:
					if isinstance(v1, (list, tuple)):
						pbounds[k1] = v1
		
		optimizer = BayesianOptimization(f=self.run_training, pbounds=pbounds, verbose=2, random_state=1)
		optimizer.maximize(init_points=10, n_iter=100)
		for i, res in enumerate(optimizer.res):
			print("Iteration {}: \n\t{}".format(i, res))
		opt_res = optimizer.max
		print(opt_res)
		return optimizer
	
	def adjust_pars(self, new_pars):
		"""
		Given a set of new parameters adjust them within the object dictionary
		:param new_pars: a dictionary of new parameters
		"""
		for key, value in new_pars.items():
			for k0, v0 in self.pars.items():
				for k1, v1 in v0.items():
					try:
						for k2, v2 in v1.items():
							if k2 == key:
								if k2 in self.int_pars:
									value = int(value)
								v1[k2] = value
					except AttributeError:
						if k1 == key:
							if k1 in self.int_pars:
								value = int(value)
							v0[k1] = value
		
	@staticmethod
	def parse_options():
		
		parser = ArgumentParser(description="This is a command line interface (CLI) for Task 4 of Speech module",
		                        epilog="Vlad Bondarenko, 2020-05-10")
		parser.add_argument("mode", action="store", type=str, metavar="<mode-to-run-in>",
		                    help="Specify a mode to run in from (train, opt, test or test_all). (default: train)")
		parser.add_argument("-c", "--config", dest="config", action="store", type=str, required=False,
		                    metavar="<path-to-file", default="Config/config.ini",
		                    help="Specify the location of config file you want to use")
		parser.add_argument("-p", "--pars", dest="pars", action="store", type=str, required=False,
		                    metavar="<path-to-file", default="Config/fnn_pars.json",
		                    help="Specify the location of parameters to use with the model.")
		parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", required=False,
		                    help="Specify if you want to print out all of the outputs and graphs")
		parser.add_argument("-s", "--save", dest="save", action="store_true", required=False,
		                    help="Specify if model should be saved")
		
		options = parser.parse_args()
		
		modes = {'train', 'test', 'opt', 'test_all'}
		if options.mode not in modes:
			parser.print_help()
			sys.exit(1)
		
		return options
		
	def load_config(self):
		"""
		Load the config from file.
	
		Parameters
		----------
		`file_location` : string
			Optionally provide the location of the config file as full absolute path. If not
			provided, config is assumed to be in 'Config/config.ini'.
		Returns
		-------
		dict
			A dictionary of config parameters whose keys match the names used in the config file.
		"""
		
		file_location = self.options.config
		
		parser = ConfigParser()
		parser.read(file_location)
		
		config = dict()
		config['general'] = dict()
		config['folders'] = dict()
		
		general = config['general']
		general['config_location'] = file_location
		general['seed'] = parser.getint("general", "seed")
		general['model_type'] = parser.get("general", "model_type")
		general['model_name'] = parser.get("general", "model_name")
		general['data_size'] = parser.getfloat("general", "data_size")
		
		folders = config['folders']
		folders['data_folder'] = parser.get("folders", "data_folder")
		folders['label_folder'] = parser.get("folders", "label_folder")
		folders['model_folder'] = parser.get("folders", "model_folder")
		
		return config
		
	def load_pars(self):
		with open(self.options.pars, 'r') as f:
			pars = json.load(f)
		return pars


if __name__ == '__main__':
	
	self = Task4()
	
	if self.options.mode == 'train':
		self.run_training()
	elif self.options.mode == 'opt':
		self.optimize()
	elif self.options.mode == 'test':
		self.run_evaluation()