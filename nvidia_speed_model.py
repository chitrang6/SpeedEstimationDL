from keras.layers.convolutional import Convolution2D # Convolutional Layer
from keras.layers.pooling import MaxPooling2D # Pooling --> Max 


from keras.layers import ELU # Non linear activation function

from keras.optimizers import Adam # Optimizer
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential


IMAGE_HEIGHT = 66
IMAGE_WIDTH  = 200
IMAGE_CHANNEL = 3



def nvidia_speed_model():
	"""Definition of the Nvidia Steering model. Here wI will use it for the Speed estimation of the car"""

	input_size = (IMAGE_HEIGHT , IMAGE_WIDTH , IMAGE_CHANNEL)
	model = Sequential()

	# Normalization 

	model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = input_size))

	model.add(Convolution2D(24, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv1'))


	model.add(ELU())

	model.add(Convolution2D(36, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv2'))
	model.add(ELU())

	model.add(Convolution2D(48, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv3'))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Convolution2D(64, (3, 3), 
                            strides = (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv4'))
	model.add(ELU())              
	model.add(Convolution2D(64, (3, 3), 
                            strides= (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv5'))
	model.add(Flatten(name = 'flatten'))
	model.add(ELU())
	model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
	model.add(ELU())
	model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
	model.add(ELU())
	model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
	model.add(ELU())
	# do not put activation at the end because we want to exact output, not a class identifier
	model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))	

	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(optimizer = adam, loss = 'mse')
	return model
