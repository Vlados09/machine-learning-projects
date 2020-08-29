import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


class GMM:
	
	"""
	Personal implementation of Gausian Mixture Model built on top of scikit-learn.
	Perform training, evaluation and optimization.
	"""
	
	def __init__(self, classes, name='', verbose=True):
		
		self.classes = classes
		self.name = name
		self.n_classes = len(classes)
		self.verbose = verbose
	
	# =================================================================
	# TRAINING ROUTINE
	# =================================================================
	
	def train(self, X, y, covariance_type='full', n_mixtures=1):
		
		if self.verbose:
			print(f'Number of training frames = {len(y)}')
		
		gmm_set = [GaussianMixture(n_components=n_mixtures, covariance_type=covariance_type,
		                           init_params='kmeans', max_iter=100) for _ in range(self.n_classes)]
		
		for c in range(self.n_classes):
			
			# Extract features for a given class
			class_features = X[y == c]
			
			if self.verbose:
				print(f'Traning {self.name} model for |{self.classes[c]}| with shape={class_features.shape}')
			
			# Train GMM
			gmm_set[c].fit(class_features)
		
		return gmm_set
	
	# =================================================================
	# EVALUATION ROUTINE
	# =================================================================
	def evaluate(self, gmm_set, X, y, utter_ids, plot='all'):
		
		# Perform classification for each frame
		log_prob = np.empty((self.n_classes, X.shape[0]))
		for c in range(self.n_classes):
			log_prob[c] = gmm_set[c].score_samples(X).flatten()
		
		# Get class with max log probability
		pred = np.argmax(log_prob, axis=0)
		
		# Compute frame level accuracy
		acc = np.mean(y == pred)
		if self.verbose:
			print(f'Number of test frames = {len(y)} \n')
			prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average='macro')
			print(f'Frame level test results for {self.name}: \n   accuracy = {acc:.3f} \n   precision = {prec:.3f} '
			      f'\n   recall = {rec:.3f} \n   f1 = {f1:.3f} \n')
			if plot == 'all' or plot == 'frame':
				mat = confusion_matrix(y, pred)
				self.plot_confusion(mat, title=f'Frame cm for {self.name}')
				self.plot_labels(X, y, pred)
		
		# Compute utterance level accuracy
		utter_acc = self.evaluate_utter(utter_ids, y, log_prob, plot=plot)
		
		return acc, utter_acc
	
	def evaluate_utter(self, utter_ids, y, log_prob, plot='all'):
		
		utter = np.array(list(set(utter_ids)))
		utter_y = np.empty_like(utter)
		utter_pred = np.empty_like(utter)
		for i, u in enumerate(utter):
			this_u = np.where(u == utter_ids)[0]
			utter_y[i] = np.mean(y[this_u])
			# Sum log probabilities of the sequence (frames):
			utter_pred[i] = np.argmax(np.sum(log_prob[:, this_u], axis=1))
		utter_acc = np.mean(utter_y == utter_pred)
		
		if self.verbose:
			prec, rec, f1, _ = precision_recall_fscore_support(utter_y, utter_pred, average='macro')
			print(f'Utterance level test results for {self.name}: \n   accuracy = {utter_acc:.3f}'
			      f'\n   precision = {prec:.3f} \n   recall = {rec:.3f} \n   f1 = {f1:.3f}')
			if plot == 'all' or plot == 'utter':
				mat = confusion_matrix(utter_y, utter_pred)
				self.plot_confusion(mat, title=f'Utterance cm for {self.name}', figsize=(8, 8))
				
		return utter_acc
	
	def plot_confusion(self, mat, title='Confusion Matrix', figsize=(5, 5)):
		# Now we can check the confusion matrix
		plt.figure(1, figsize=figsize)
		sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
		            xticklabels=self.classes, yticklabels=self.classes, cmap="YlGnBu")
		plt.xlabel(f'predicted {self.name} label')
		plt.ylabel(f'target {self.name} label')
		plt.title(title)
		
		plt.show()

	def plot_labels(self, X, y, pred):
		
		# Now we can scatter plot MFCCs. We use C2 and C3 instead of C1 and C2 as the
		# scatter plot looks better
		f1 = 1  # C2
		f2 = 2  # C3
		plt.figure(2)
		plt.subplot(121)
		plt.scatter(X[:, f1], X[:, f2], s=30, c=y, cmap='viridis', alpha=0.8)
		plt.title(f'target {self.name} labels')
		plt.xlabel('C{}'.format(f1 + 1))
		plt.ylabel('C{}'.format(f2 + 1))
		
		plt.subplot(122)
		plt.scatter(X[:, f1], X[:, f2], s=30, c=pred, cmap='viridis', alpha=0.8)
		plt.title(f'predicted {self.name} labels')
		plt.xlabel('C{}'.format(f1 + 1))
		plt.ylabel('C{}'.format(f2 + 1))
		
		plt.show()
		
	# =================================================================
	# HYPER-PARAMETER SEARCH
	# =================================================================

	def optimize(self, X, y, utter_ids, mixtures_range=np.arange(1, 11), dev_split=0.1, n_repeats=10):
		
		# Split train data into train and development set
		X_train, X_dev, y_train, y_dev, utter_tr, utter_dev = train_test_split(X, y, utter_ids,
		                                                                       test_size=dev_split, shuffle=True)
		
		self.verbose = False  # Disable printing for optimization
		covariance_types = ['full', 'tied', 'diag', 'spherical']
		
		fig1, ax1 = plt.subplots(2, 2, sharey='all')
		ax1[0, 0].set_title(f'Average dev accuracy for {self.name} over {n_repeats} repeats')
		fig1.set_figwidth(12)
		fig1.set_figheight(8)
		
		fig2, ax2 = plt.subplots(1)
		
		for i, cov_t in enumerate(covariance_types):
			cov_acc = []
			cov_iter = []
			for n in mixtures_range:
				accuracy = 0
				utter_accuracy = 0
				n_iter = 0
				for _ in range(n_repeats):
					# Train GMM with chose parameters
					gmm_set = self.train(X_train, y_train, covariance_type=cov_t, n_mixtures=n)
					for g in gmm_set:
						n_iter += g.n_iter_
					# Evaluate on dev set
					acc, utter_acc = self.evaluate(gmm_set, X_dev, y_dev, utter_dev)
					accuracy += acc
					utter_accuracy += utter_acc
				cov_acc.append((accuracy/n_repeats, utter_accuracy/n_repeats))
				cov_iter.append(n_iter/n_repeats)
			a, a_u = list(zip(*cov_acc))
			# Plot results separetly for each type of covariance:
			ax1[i % 2, int(i/2)].plot(a, label=f'{cov_t} frame')
			ax1[i % 2, int(i / 2)].plot(a_u, label=f'{cov_t} utter')
			ax2.plot(cov_iter, label=cov_t)
		
		for ax in fig1.get_axes():
			ax.set_ylabel(f'Accuracy')
			ax.set_xlabel('Number of Mixtures')
			ax.label_outer()
			ax.legend()

		ax2.set_title(f'Convergence Optimization for {self.name}')
		ax2.set_xlabel('Number of Mixtures')
		ax2.set_ylabel(f'Average Iterations over {n_repeats} repetitions')
		ax2.legend()
		
		plt.show()
		
		self.verbose = True  # Enable printing