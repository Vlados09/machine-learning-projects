import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L1L2


class VADModel:
	
	def __init__(self, config, pars, options, input_shape):
		
		self.config = config
		self.pars = pars
		self.options = options
		self.input_shape = input_shape
		
		self.model_folder = config['folders']['model_folder']
		self.model_type = config['general']['model_type']
		self.model_name = config['general']['model_name']
		self.model = self.init_model()
		
	def init_model(self):
		"""
		Initialise a model
		:return: keras model
		"""
		model_switcher = {
			'FNN': self.FNN(),
			'LSTM': self.LSTM()
		}
		return model_switcher[self.model_type]
	
	def train(self, X_train, y_train, X_dev, y_dev):
		history = self.model.fit(X_train, y_train, validation_data=(X_dev, y_dev),
		                         **self.pars[self.model_type]['train'])
		if self.options.verbose:
			self.plot_history(history)
	
	def predict(self, X):
		batch_size = self.pars[self.model_type]['train']['batch_size']
		verbose = self.pars[self.model_type]['train']['verbose']
		return self.model.predict(X, batch_size=batch_size, verbose=verbose)
	
	def save(self):
		self.model.save(f'{self.model_folder}{self.model_name}')
		
	def load(self):
		try:
			self.model = load_model(f'{self.model_folder}{self.model_name}')
		except FileNotFoundError:
			raise Exception(f'Mode {self.model_name} has not been saved.')
	
	@staticmethod
	def plot_history(history):
		plt.plot(history.history['loss'], label='train loss')
		plt.plot(history.history['val_loss'], label='dev loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.show()
	
	def FNN(self):
		try:
			pars = self.pars['FNN']['model']
			ker_reg = L1L2(l1=pars['ker_reg1'], l2=pars['ker_reg2'])
			model = Sequential([
				Dense(pars['l1'], activation='relu', input_shape=(self.input_shape[1],), kernel_regularizer=ker_reg),
				Dropout(pars['d1']),
				# Dense(pars['l2'], activation='relu', kernel_regularizer=ker_reg),
				# Dropout(pars['d2']),
				Dense(1, activation='sigmoid')
			])
			model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=pars['lr']))
			self.pars['FNN']['train']['callbacks'] = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1)]
		except KeyError:
			model = None
		return model
	
	def LSTM(self):
		try:
			pars = self.pars['LSTM']['model']
			reg = L1L2(l1=pars['ker_reg1'], l2=pars['ker_reg2'])
			model = Sequential([
				LSTM(pars['l1'], activation='relu', input_shape=(self.input_shape[1], self.input_shape[2]),
				                kernel_regularizer=reg, recurrent_regularizer=reg),
				Dropout(pars['d1']),
				Dense(pars['l2'], activation='relu', kernel_regularizer=reg),
				Dropout(pars['d2']),
				Dense(1, activation='sigmoid')
			])
			model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=pars['lr']))
			self.pars['LSTM']['train']['callbacks'] = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1)]
		except KeyError:
			model = None
		return model