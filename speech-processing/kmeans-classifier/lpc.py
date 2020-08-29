# This script computes the MFCC features for automatic speech recognition
#
# You need to complete the part indicated by #### so that the code can produce
# sensible results.
#
# Ning Ma (n.ma@sheffield.ac.uk)
#


import numpy as np
import scipy.io.wavfile as wav
from scipy.linalg import toeplitz
from scipy.fftpack import dct, fft, ifft


class LPC:
	
	def __init__(self, file, use_hamming=True, use_preemph=False):
		
		self.fs_hz, self.signal = wav.read(file)
		self.signal_length = len(self.signal)
		
		# Define parameters
		self.order = int(((self.fs_hz/1000) * 2) + 2)
		self.use_preemph = use_preemph       # where as to use pre-emphasis or not
		self.preemph = 0.97                  # pre-emphasis coefficient
		self.frame_length_ms = 25            # frame length in ms
		self.frame_step_ms = 10              # frame shift in ms
		self.eps = 0.001                     # Floor to avoid log(0)
		self.use_hamming = use_hamming       # Determines if hamming should be used or not
		
	def get_lpc(self):
		
		# Pre-emphasis
		if self.use_preemph:
			signal = self.pre_emphasize(self.signal)
		else:
			signal = self.signal
		
		# Split signal into frames
		frames = self.split_frames(signal)
		frame_length = frames.shape[1]
		
		# Apply the Hamming window
		if self.use_hamming:
			frames *= np.hamming(frame_length)
		
		coeff = np.empty((frames.shape[0], self.order+1))
		corr = np.empty((frames.shape[0], self.order+1, self.order+1))
			
		for i, frame in enumerate(frames):
			r = self.auto_correlate(frame)
			corr[i, :, :] = toeplitz(r)
			a, e, k = self.levinson_lpc(r)
			coeff[i, :] = a
		
		return coeff, corr

	def pre_emphasize(self, signal):
		"""
			Applies pre-emphasis filter on the signal to amplify the high frequencies.
			Using: H(t) = x(t) - preemph * x(t-1)

			----------
			Parameters
			----------
			signal : array (n)
				Raw audio signal
			
			Returns
			-------
			signal : array (n-1)
				Pre-emphasized signal
		"""
		
		return np.append(signal[0], signal[1:] - self.preemph * signal[:-1])
	
	def split_frames(self, signal):
		"""
			Splits a signal into equal size frames with a step.
			Uses frame length and step defined during initialization
			
			----------
			Parameters
			----------
			signal : array (n)
				Signal to be split
			
			Returns
			-------
			frames : array (num_frames * frame_length)
			
		"""
		# Compute number of frames and padding
		frame_length = int(round(self.frame_length_ms / 1000.0 * self.fs_hz))
		frame_step = int(round(self.frame_step_ms / 1000.0 * self.fs_hz))
		num_frames = int(np.ceil(float(self.signal_length - frame_length) / frame_step))
		print("number of frames is {}".format(num_frames))
		pad_signal_length = num_frames * frame_step + frame_length
		pad_zeros = np.zeros((pad_signal_length - self.signal_length))
		pad_signal = np.append(signal, pad_zeros)
		
		# Split signal into frames
		frames = np.empty((num_frames, frame_length))
		for i in range(num_frames):
			start = i*frame_step
			frames[i] = pad_signal[start:start+frame_length]
		
		return frames
	
	def auto_correlate(self, frame):
		R = [np.dot(frame, frame)]
		for i in range(1, self.order+1):
			r = np.dot(frame[i:], frame[:-i])
			R.append(r)
		R = np.array(R) / frame.size
		return R
	
	def levinson_lpc(self, r):
		
		a = np.empty(self.order + 1)
		t = np.empty_like(a)
		k = np.empty_like(a)
		
		a[0] = 1
		e = r[0]
		
		for i in range(1, self.order+1):
			acc = r[i]
			for j in range(1, i):
				acc += a[j] * r[i-j]
			k[i-1] = -acc/e
			a[i] = k[i-1]
			
			for j in range(self.order):
				t[j] = a[j]
				
			for j in range(1, i):
				a[j] += k[i-1] * np.conj(t[i-j])
			
			e *= 1 - k[i-1] * np.conj(k[i-1])
			
		return a, e, k
		
		
if __name__ == '__main__':

	self = LPC('SA1.wav')
