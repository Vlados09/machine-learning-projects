# This script performs K-means clustering
#
# Ning Ma (n.ma@sheffield.ac.uk)
# Thomas Hain 
#
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode
from my_mfcc import MFCC
from lpc import LPC

# ==================================================
# CLUSTERING ROUTINES
# ==================================================


def euclidean(X, y):
	"""
		Calculate the Euclidean distance
	"""
	return np.sqrt(np.sum((X - y) ** 2, axis=1))


def mahalanobis(X, y, inv_cov=None):
	"""
		Compute the Mahalanobis Distance between each row of x and the data y
	"""
	if inv_cov is None:
		# Equivalent to euclidean
		inv_cov = np.eye(X.shape[1])
	diff = X - y
	d_2 = np.dot(np.dot(diff, inv_cov), diff.T)
	return d_2.diagonal()


def itakura(X, y, R=None):
	"""
		Computer the itakura distance between all data points in X and point y from X.
	"""
	n_samples = X.shape[0]
	dI = np.empty(n_samples)
	for i in range(n_samples):
		# Errors for filter a and b:
		e_a = np.dot(np.dot(X[i], R[i]), X[i])
		e_b = np.dot(np.dot(y, R[i]), y)
		# Final Itakura distance measure:
		dI[i] = -np.log(e_a / e_b)
		
	return dI


def kmeans_clustering(X, n_classes, init='k-means++', dist='euclidean', R=None, rseed=1, plot=False):
	"""
	K-Means clustering

	Parameters:
	-----------
		X: data to be clustered [num_samples x num_features]
		num_clusters: number of clusters
		rseed: random seed (default 1)

	Returns:
	-------
		clusters: clustered labels for each sample in X
		centres: cluster centres
	"""
	
	# 1. Randomly choose num_clusters samples as cluster means
	np.random.seed(rseed)
	
	centers = init_centers(X, n_classes, init)
	new_centers = np.empty_like(centers)
	
	# Select the distance metric:
	if dist == 'euclidean':
		partial_dist = partial(euclidean, X)
	elif dist == 'mahalanobis':
		inv_cov = np.linalg.pinv(np.cov(X.T))
		partial_dist = partial(mahalanobis, X, inv_cov=inv_cov)
	elif dist == 'itakura':
		partial_dist = partial(itakura, X, R=R)
	else:
		raise Exception('Incorrect distance provided, should be one of {euclidean, mahalanobis, itakura}')
	
	while True:
		# 2a. Assign data based on closest centre
		distances = np.array(list(map(partial_dist, centers)))
		clusters = np.argmin(distances, axis=0)
		
		if plot:
			plt.scatter(X[:, 1], X[:, 2], s=30, c=clusters, cmap='viridis', alpha=0.8)
			plt.scatter(centers[:, 1], centers[:, 2], marker='*', s=100, c='red')
			plt.show()

		# 2b. Update cluster centres from means of each cluster
		for i in range(n_classes):
			x = X[np.where(clusters == i)]
			if len(x):
				new_centers[i, :] = x.mean(axis=0)
		
		# Check for convergence
		if np.all(centers == new_centers):
			break
		centers = new_centers.copy()

	return clusters, centers


def init_centers(X, n_classes, init):
	"""
	Initialize the centers of clusters using either random or k-means++ algorithms:
	"""
	
	n_samples = X.shape[0]
	rand_choice = partial(np.random.choice, range(n_samples))
	part_dist = partial(euclidean, X)
	
	if init == 'k-means++':
		rand_idx = []
		# First choice is selected randomly with uniform probabilities
		p = np.random.uniform(size=n_samples)
		for _ in range(n_classes):
			r = rand_choice(p=p/np.sum(p))
			rand_idx.append(r)
			# Next probability distribution is proportional to min distance squared:
			p = np.min(list(map(part_dist, X[rand_idx])), axis=0) ** 2
	else:
		rand_idx = rand_choice(n_classes, replace=False)
		
	centers = X[rand_idx]
	
	return centers


def sklearn_clustering(X, n_classes, random_state=0):
	"""
		Perform clustering using sklearn KMeans
	"""
	kmeans = KMeans(n_classes, random_state=random_state, max_iter=1, algorithm='full')
	clusters = kmeans.fit_predict(X)
	return clusters


def evaluate_clustering(X, y, n_classes, n_iter=10, dist='euclidean', R=None):
	"""
		Evaluate and compare clustering approaches over several repetitions.
	"""
	base_f1 = []
	test_f1 = []
	
	for i in range(n_iter):
		
		rseed = np.random.randint(10000)
		
		base_clusters = sklearn_clustering(X, n_classes, random_state=rseed)
		base_labels = fix_labels(base_clusters, y, n_classes)
		base_f1.append(f1_score(y, base_labels, average='macro'))
		
		test_clusters, _ = kmeans_clustering(X, n_classes, rseed=rseed, dist=dist, R=R)
		test_labels = fix_labels(test_clusters, y, n_classes)
		f1 = f1_score(y, test_labels, average='macro')
		test_f1.append(f1)
		
		# print(f'Evaluation step {i} with random seed={rseed} and F1={f1_score(y, test_labels, average="macro")}')
	
	print(f'Scikit-Learn F1 score: mean={np.mean(base_f1):.2f}, var={np.var(base_f1):.2f}')
	print(f'Personal F1 score: mean={np.mean(test_f1):.2f}, var={np.var(test_f1):.2f}')
		
		
def fix_labels(clusters, y, n_classes):
	# Because k-means knows nothing about the identity of the cluster, the labels
	# may be permuted. We can fix this by matching each learned cluster label with
	# the true labels
	
	labels = np.zeros_like(clusters)
	for n in range(n_classes):
		mask = (clusters == n)
		labels[mask] = mode(y[mask])[0]
	
	return labels

# ==================================================
# COMPUTE MFCCs/LPCs for a list of vowels
# ==================================================


def get_data(vowels, feature='mfcc', scale=False):
	"""
		Get either MFCC or LPC features with labels
	"""
	
	X = np.array([])
	y = []
	R = np.array([])
	
	for i, vowel in enumerate(vowels):
		
		print(f'Processing vowel {vowel}')
		file = f'vowels/{vowel}.wav'
		
		if feature == 'mfcc':
			# Get MFCC features:
			x = MFCC(file, deltas=False).get_mffc()
		elif feature == 'lpc':
			# Get Linear Prediction Coefficients and related autocorrelation matrix R
			x, r = LPC(file, use_preemph=False).get_lpc()
			R = np.concatenate((R, r)) if R.size else r
		else:
			raise Exception('Incorrect feature selected. Should be one of {mfcc, lpc}')
		
		num_frames = x.shape[0]
		X = np.concatenate((X, x)) if X.size else x
		y = np.append(y, np.tile(i, num_frames))
	
	if scale:
		X = MinMaxScaler().fit_transform(X)
	
	return X, y, R


# ==================================================
# PLOTTING
# ==================================================


def plot_results(X, y, clusters, n_classes, vowels, title='Confusion Matrix'):
	
	labels = fix_labels(clusters, y, n_classes)
	
	# Now we can check the confusion matrix
	mat = confusion_matrix(y, labels)
	plt.figure(1)
	seaborn.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
					xticklabels=vowels, yticklabels=vowels, cmap="YlGnBu")
	plt.title(title)
	plt.xlabel('predicted vowel label')
	plt.ylabel('target vowel label')
	plt.show()
	
	# Now we can scatter plot MFCCs. We use C2 and C3 instead of C1 and C2 as the
	# scatter plot looks better
	f1 = 1  # C2
	f2 = 2  # C3
	plt.figure(2)
	plt.subplot(121)
	plt.scatter(X[:, f1], X[:, f2], s=30, c=y, cmap='viridis', alpha=0.8)
	plt.title('true vowel labels')
	plt.xlabel('C{}'.format(f1 + 1))
	plt.ylabel('C{}'.format(f2 + 1))
	
	plt.subplot(122)
	plt.scatter(X[:, f1], X[:, f2], s=30, c=labels, cmap='viridis', alpha=0.8)
	plt.title('predicted vowel labels')
	plt.xlabel('C{}'.format(f1 + 1))
	plt.ylabel('C{}'.format(f2 + 1))
	
	plt.show()
	
	
def run_task(feature, dist, scale=False):
	
	print(f'Running task 6 with {feature} features and {dist} distance')
	
	vowels = ['a', 'e', 'i', 'u']
	n_classes = len(vowels)
	
	X, y, R = get_data(vowels, feature=feature, scale=scale)
	
	sk_clusters = sklearn_clustering(X, n_classes, random_state=123)
	plot_results(X, y, sk_clusters, n_classes, vowels, title='scikit-learn cm')
	
	my_clusters, centers = kmeans_clustering(X, n_classes, rseed=123, dist=dist, R=R, plot=True)
	plot_results(X, y, my_clusters, n_classes, vowels, title='personal cm')
	
	evaluate_clustering(X, y, n_classes, n_iter=100, R=R, dist=dist)


if __name__ == '__main__':
	run_task('lpc', 'itakura', scale=False)

	
	
	
	


