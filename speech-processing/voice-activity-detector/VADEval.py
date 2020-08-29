import numpy as np
import matplotlib.pyplot as plt
import seaborn

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve


class VADEval:
	
	def __init__(self, config, options):
		self.config = config
		self.options = options
		self.model_name = config['general']['model_name']
	
	def evaluate(self, y_true, y_prob, split='dev'):
		
		# Calculate EER and where it occurs (threshold):
		fpr, tpr, thresh = roc_curve(y_true, y_prob)
		fnr = 1 - tpr
		eer, eer_thresh, eer_idx = self.eer_score(fpr, fnr, thresh)
	
		if self.options.verbose:
			
			# Convert probability to class prediction:
			y_pred = (y_prob > 0.5).astype('int')
			accuracy = accuracy_score(y_true, y_pred)
			
			print(f'VAD prediction results for {self.model_name}: '
			      f'\n   accuracy = {accuracy:.2f}',
			      f'\n   EER = {eer:.2f}')
			
			# Compute baseline DET curve:
			base_prob = np.zeros_like(y_true)
			base_fpr, base_tpr, _ = roc_curve(y_true, base_prob)
			base_fnr = 1 - base_tpr
			
			# Plot results:
			self.plot_det(fpr, fnr, base_fpr, base_fnr, eer_idx)
			self.plot_confusion(y_true, y_pred)
		
		return eer
		
	@staticmethod
	def eer_score(fpr, fnr, thresh):
		"""
		Compute EER score
		:param fpr: false positive rations
		:param fnr: false negative rations
		:param thresh: threshold location
		:return: EER, EER threshold value, EER index location
		"""
		eer_idx = np.argmin(np.abs(fnr - fpr))
		eer_thresh = thresh[eer_idx]
		eer = fpr[eer_idx]
		
		return eer, eer_thresh, eer_idx
	
	def plot_det(self, fpr, fnr, base_fpr, base_fnr, eer_idx):
		
		plt.figure()
		plt.plot(fpr, fnr, label=self.model_name)
		plt.plot(base_fpr, base_fnr, c='b', linestyle='--', label='baseline')
		plt.scatter(fpr[eer_idx], fnr[eer_idx], s=30, c='r', label='EER Threshold')
		plt.title(f'DET Curve for {self.model_name}')
		plt.xlabel('False Positive Rate')
		plt.ylabel('Missed Detection Rate')
		plt.legend(loc='upper right')
		plt.show()
	
	def plot_confusion(self, y_true, y_pred):
		cm = confusion_matrix(y_true, y_pred)
		plt.figure()
		seaborn.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, cmap="YlGnBu")
		plt.xlabel(f'Predicted VA label')
		plt.ylabel(f'Target VA label')
		plt.title(f'Confusion Matrix for {self.model_name}')
		plt.show()
			
	
	

