# This script computes the MFCC features for automatic speech recognition
#
# You need to complete the part indicated by #### so that the code can produce
# sensible results.
#
# Ning Ma (n.ma@sheffield.ac.uk)
#

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import dct


class MFCC:
	
	def __init__(self, file, num_ceps=12, use_hamming=True, num_filters=26, use_preemph=True):
		
		self.fs_hz, self.signal = wav.read(file)
		self.signal_length = len(self.signal)
		
		# Define parameters
		self.use_preemph = use_preemph       # where as to use pre-emphasis or not
		self.preemph = 0.97                  # pre-emphasis coefficient
		self.frame_length_ms = 25            # frame length in ms
		self.frame_step_ms = 10              # frame shift in ms
		self.low_freq_hz = 0                 # filterbank low frequency in Hz
		self.high_freq_hz = 8000             # filterbank high frequency in Hz
		self.nyquist = self.fs_hz / 2.0      # Check the Nyquist frequency
		if self.high_freq_hz > self.nyquist:
			self.high_freq_hz = self.nyquist
		self.num_filters = num_filters       # number of mel-filters
		self.num_ceps = num_ceps             # number of cepstral coefficients (excluding C0)
		self.cep_lifter = 22                 # Cepstral liftering order
		self.eps = 0.001                     # Floor to avoid log(0)
		self.use_hamming = use_hamming       # Determines if hamming should be used or not
		
	def get_mffc(self, verbose=True):
		
		# Pre-emphasis
		if self.use_preemph:
			signal = self.pre_emphasize(self.signal)
		else:
			signal = self.signal
		
		# Split signal into frames
		frames = self.split_frames(signal)
		frame_length = frames.shape[1]
		
		# Find the smallest power of 2 greater than frame_length
		NFFT = 1 << (frame_length - 1).bit_length()
		
		# Apply the Hamming window
		if self.use_hamming:
			frames *= np.hamming(frame_length)
		
		# Compute magnitude spectrum
		magspec = np.absolute(np.fft.rfft(frames, NFFT))
		
		# Compute power spectrum
		powspec = np.power(magspec, 2) / NFFT
		
		# Compute mel-filters
		mel_filters = self.compute_mel_filters(NFFT)
	
		# Compute log mel spectrum
		fbank = np.dot(powspec, mel_filters.T)
		fbank[fbank < self.eps] = self.eps
		fbank = np.log(fbank)
		
		# Apply DCT to get num_ceps MFCCs, omit C0
		mfcc = dct(fbank, norm='ortho')[:, 1:self.num_ceps+1]
	
		# Liftering
		mfcc = self.liftering(mfcc)
		
		# Log-compress power spectrogram
		powspec = np.log(powspec + self.eps)
		
		if verbose:
			print("=== Before normalisation")
			print(f"mfcc mean = {np.round(np.mean(mfcc, axis=0), 2)}")
			print(f"mfcc std = {np.round(np.std(mfcc, axis=0), 2)}")
		
		mfcc_z = (mfcc - np.mean(mfcc, axis=0))/np.std(mfcc, axis=0)
		
		if verbose:
			print("=== After normalisation")
			print(f"mfcc mean = {np.round(np.mean(mfcc_z, axis=0), 2)}")
			print(f"mfcc std = {np.round(np.std(mfcc_z, axis=0), 2)}")
		
		if verbose:
			self.plot_results(powspec, fbank, mfcc, mfcc_z)
			
		return mfcc_z
		
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
	
	def compute_mel_filters(self, NFFT):
		"""
			Computes mel filter banks

			----------
			Parameters
			----------
			NFFT : int

			Returns
			-------
			mel_filter : array (NFFT//2 + 1 * num_filters)

		"""
		mel_filters = np.zeros((self.num_filters, NFFT // 2 + 1))
		
		# Calculate filter centers
		low_freq_mel = self.freq2mel(self.low_freq_hz)
		high_freq_mel = self.freq2mel(self.high_freq_hz)
		mel_centers = np.linspace(low_freq_mel, high_freq_mel, self.num_filters+2)
		hz_centers = self.mel2freq(mel_centers)
		idx = ((hz_centers * NFFT+1) // self.fs_hz).astype(np.int)
		
		for i in range(0, self.num_filters):
			# For each filter perform linear interpolation between edges and center:
			for j in range(idx[i], idx[i + 1]):
				mel_filters[i, j] = (j - idx[i]) / (idx[i + 1] - idx[i])
			for j in range(idx[i + 1], idx[i + 2]):
				mel_filters[i, j] = (idx[i + 2] - j) / (idx[i + 2] - idx[i + 1])
		
		return mel_filters
	
	def liftering(self, mfcc):
		"""
			Applies sinusoidal liftering to de-emphasize higher MFCCs
			----------
			Parameters
			----------
			mfcc :  np.array

			Returns
			-------
			mfcc : np.array
				with lift applied

		"""
		lift = 1 + (self.cep_lifter / 2.0) * np.sin(np.pi * np.arange(self.num_ceps) / self.cep_lifter)
		mfcc *= lift
		return mfcc
		
	@staticmethod
	def freq2mel(freq):
		"""Convert Frequency in Hertz to Mels
	
		Args:
			freq: A value in Hertz. This can also be a numpy array.
	
		Returns
			A value in Mels.
		"""
		return 2595 * np.log10(1 + freq / 700.0)
	
	@staticmethod
	def mel2freq(mel):
		"""Convert a value in Mels to Hertz
	
		Args:
			mel: A value in Mels. This can also be a numpy array.
	
		Returns
			A value in Hertz.
		"""
		return 700 * (10 ** (mel / 2595.0) - 1)
	
	@staticmethod
	def write_to_txt(mfcc):
		with open("mfcc.txt", "w") as txt_file:
			for vec in mfcc.T:
				txt_file.write(" ".join(np.round(vec, 3).astype('str')) + "\n")
	
	def plot_results(self, powspec, fbank, mfcc, mfcc_z):
		# Plotting power spectrogram vs mel-spectrogram
		plt.figure(1)
		siglen = len(self.signal) / np.float(self.fs_hz);
		plt.subplot(211)
		plt.imshow(powspec.T, origin='lower', aspect='auto', extent=(0, siglen, 0, self.fs_hz / 2000), cmap='jet')
		plt.title('Power Spectrogram')
		plt.gca().get_xaxis().set_ticks([])
		plt.gca().set_yticklabels(['', 1, 2, 3, 4, 5, 6, 7, 8])
		plt.ylabel('Frequency (kHz)')
		
		plt.subplot(212)
		# freq_bins = freq_bins.astype(int)
		plt.imshow(fbank.T, origin='lower', aspect='auto', extent=(0, siglen, 0, self.num_filters), cmap='jet')
		plt.yticks(np.arange(0, fbank.shape[1]+1, int(fbank.shape[1]/5)))
		# plt.gca().set_yticklabels(['', freq_bins[5], freq_bins[10], freq_bins[16], freq_bins[21], freq_bins[27]])
		plt.title('Mel-filter Spectrogram')
		plt.xlabel('Time (s)')
		plt.ylabel('Frequency (Hz)')
		
		plt.show()
		
		# Plotting MFCCs with CMN
		plt.figure(2)
		plt.subplot(211)
		plt.imshow(mfcc.T, origin='lower', aspect='auto', extent=(0, siglen, 1, self.num_ceps), cmap='jet')
		plt.title('MFCC without mean and variance normalisation')
		
		plt.subplot(212)
		plt.imshow(mfcc_z.T, origin='lower', aspect='auto', extent=(0, siglen, 1, self.num_ceps), cmap='jet')
		plt.title('MFCC with mean and variance normalisation')
		
		plt.show()
		
		
if __name__ == '__main__':

	print('\n Default MFCC set up:')
	MFCC_ = MFCC('SA1.wav')
	mfcc_ = MFCC_.get_mffc(verbose=True)
	MFCC_.write_to_txt(mfcc_)
	
	print('\n No Hamming Window:')
	MFCC_a = MFCC('SA1.wav', use_hamming=False)
	MFCC_a.get_mffc(verbose=False)
	
	print('\n Increases Number of MFCCs to 40:')
	MFCC_b = MFCC('SA1.wav', num_filters=41, num_ceps=40)
	MFCC_b.get_mffc(verbose=False)
	
	print('\n 80 mel filter banck and 40 MFCCs: ')
	MFCC_c = MFCC('SA1.wav', num_filters=80, num_ceps=40)
	MFCC_c.get_mffc(verbose=False)
	
	print('\n No pre-emphasis applied:')
	MFCC_d = MFCC('SA1.wav', use_preemph=False)
	MFCC_d.get_mffc(verbose=False)
