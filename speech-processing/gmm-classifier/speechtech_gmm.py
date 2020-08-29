# This script performs frame-based vowel and speaker classification using
# Gaussian Mixture Models

# =================================================================
# GENERAL IMPORTS
# =================================================================
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from my_mfcc import MFCC
from my_gmm import GMM

sns.set()  # for plot styling

# =================================================================
# DATA EXTRACTION
# =================================================================


def get_vowel_data():
	
	classes = ['a', 'e', 'i', 'u']
	
	# Load all vowel signals and compute MFCCs
	# Accumulate MFCCs for all frames and corresponding labels
	features = np.array([])
	target_labels = []
	for vowel_id in range(len(classes)):
		print(f'Processing vowel {classes[vowel_id]}')
		# Read waveform
		mfcc = MFCC(f'vowels/{classes[vowel_id]}.wav').get_mffc()
		num_frames = mfcc.shape[0]
		features = np.concatenate((features, mfcc)) if features.size else mfcc
		target_labels = np.append(target_labels, np.tile(vowel_id, num_frames))
	
	return features, target_labels, classes


def get_speaker_data(feature='mfc', scale=False):
	
	"""
	Extract speaker data for a given feature type.
	"""
	
	idx = ['spkrID', 'utterID']
	
	# Determine how many features are in the file
	n_features = pd.read_csv(f'speakers/20spkrs_{feature}_test.dat', delimiter=' ').shape[1] - len(idx)
	names = idx + [f'feature {i}' for i in range(n_features)]
	
	# Load all the partitions
	speaker_data = []
	for split in ['train', 'test']:
		speaker_split = pd.read_csv(f'speakers/20spkrs_{feature}_{split}.dat', delimiter=' ', names=names,
		                            index_col=idx)
		speaker_data.append(speaker_split)
	
	# Get labels and utterance ids
	X_train, X_test = speaker_data
	y_train = X_train.index.get_level_values(0).values
	y_test = X_test.index.get_level_values(0).values
	train_utters = X_train.index.get_level_values(1).values
	test_utters = X_test.index.get_level_values(1).values
	
	if scale:
		# Scale between 0 and 1
		scaler = MinMaxScaler().fit(X_train)
		X_train = pd.DataFrame(data=scaler.transform(X_train))
		X_test = pd.DataFrame(data=scaler.transform(X_test))
	
	return X_train.values, X_test.values, y_train, y_test, train_utters, test_utters

# =================================================================
# TRAINING AND EVALUATION
# =================================================================


def train_evaluate_vowels():
	# Training and evaluation for vowels data:
	X, y, vowels = get_vowel_data()
	vowel_gmm = GMM(vowels, name='vowels', verbose=True)
	vowel_gmm.optimize(X, y, y, n_repeats=10, mixtures_range=np.arange(1, 8))
	# Train GMMs
	vowel_gmm_set = vowel_gmm.train(X, y, n_mixtures=1, covariance_type='full')
	# NOTE: WE ARE TESTING THE GMMS USING THE TRAINING SET
	vowel_gmm.evaluate(vowel_gmm_set, X, y, utter_ids=y, plot='frame')
	
	
def train_evaluate_speaker(feature='form', scale=False):
	# Trainining and evaluation for speaker data:
	X_train, X_test, y_train, y_test, utter_train, utter_test = get_speaker_data(feature=feature, scale=scale)
	speaker_gmm = GMM(list(set(y_test)), name=f'speakers_{feature}', verbose=True)
	speaker_gmm.optimize(X_train, y_train, utter_train, mixtures_range=np.arange(1, 6), dev_split=0.2, n_repeats=1)
	speaker_gmm_set = speaker_gmm.train(X_train, y_train, n_mixtures=3, covariance_type='full')
	speaker_gmm.evaluate(speaker_gmm_set, X_test, y_test, utter_ids=utter_test, plot='utter')


def main():
	
	train_evaluate_vowels()
	# train_evaluate_speaker(feature='form')
	# train_evaluate_speaker(feature='mfb', scale=True)
	train_evaluate_speaker(feature='mfc')
	
# =================================================================


if __name__ == '__main__':
	main()

