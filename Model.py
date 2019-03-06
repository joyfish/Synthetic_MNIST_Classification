
from tensorflow.python.keras import models , optimizers , losses ,activations , callbacks
from tensorflow.python.keras.layers import *
from PIL import Image
import tensorflow as tf
import time
import os
import numpy as np
import winsound


class Classifier (object) :

	def __init__( self , number_of_classes ):

		dropout_rate = 0.5
		self.__DIMEN = 28
		self.__num_channels = 3

		input_shape = ( (self.__DIMEN**2) * self.__num_channels , )
		convolution_shape = ( self.__DIMEN , self.__DIMEN , self.__num_channels )
		kernel_size_1 = ( 4 , 4 )
		kernel_size_2 = ( 3 , 3 )
		pool_size_1 = ( 3 , 3 )
		pool_size_2 = ( 2 , 2 )
		strides = 1

		self.__NEURAL_SCHEMA = [

			Reshape( input_shape=input_shape , target_shape=convolution_shape),

			Conv2D( 32, kernel_size=kernel_size_1 , strides=strides , activation=activations.leaky_relu ),
			Conv2D( 32, kernel_size=kernel_size_1, strides=strides  , activation=activations.leaky_relu ),
			MaxPooling2D(pool_size=pool_size_1, strides=strides ),

			Conv2D( 64, kernel_size=kernel_size_2 , strides=strides , activation=activations.leaky_relu ),
			Conv2D( 64, kernel_size=kernel_size_2, strides=strides  , activation=activations.leaky_relu ),
			MaxPooling2D(pool_size=pool_size_2 , strides=strides),

			Flatten(),

			Dense( 128 , activation=activations.leaky_relu ) ,
			Dropout(dropout_rate),

			Dense( number_of_classes, activation=tf.nn.softmax )

		]

		self.__model = tf.keras.Sequential( self.__NEURAL_SCHEMA )
		self.__model.compile(
			optimizer=optimizers.Adam( lr=0.001 ),
			loss=losses.categorical_crossentropy ,
			metrics=[ 'accuracy' ] ,
		)

	def fit(self, X, Y ,  hyperparameters  ):
		initial_time = time.time()
		if not hyperparameters[ 'callbacks'] is None:
			hyperparameters[ 'callbacks' ] += [ EpochEndCallback() ]
		self.__model.fit( X  , Y ,
						 batch_size=hyperparameters[ 'batch_size' ] ,
						 epochs=hyperparameters[ 'epochs' ] ,
						 callbacks=hyperparameters[ 'callbacks'],
						 validation_data=hyperparameters[ 'val_data' ]
						 )
		final_time = time.time()
		eta = ( final_time - initial_time )
		time_unit = 'seconds'
		if eta >= 60 :
			eta = eta / 60
			time_unit = 'minutes'
		self.__model.summary( )
		print( 'Elapsed time acquired for {} epoch(s) -> {} {}'.format( hyperparameters[ 'epochs' ] , eta , time_unit ) )

	def prepare_images_from_dir( self , dir_path , num_channels ) :
		images = list()
		images_names = os.listdir( dir_path )
		for imageName in images_names :
			if num_channels == 1 :
				image = Image.open(dir_path + imageName).convert( 'L' )
			else :
				image = Image.open(dir_path + imageName)
			resize_image = image.resize((self.__DIMEN, self.__DIMEN))
			print( dir_path + imageName )
			array = list()
			for x in range(self.__DIMEN):
				sub_array = list()
				for y in range(self.__DIMEN):
					sub_array.append(resize_image.load()[x, y])
				array.append(sub_array)
			image_data = np.array(array)
			image = np.array(np.reshape(image_data,(self.__DIMEN, self.__DIMEN, self.__num_channels)))
			images.append(image)

		return np.array( images )

	def evaluate(self , test_X , test_Y  ) :
		return self.__model.evaluate(test_X, test_Y)

	def predict(self, X  ):
		predictions = self.__model.predict( X  )
		return predictions

	def save_model(self , file_path ):
		self.__model.save(file_path )

	def load_model(self , file_path ):
		self.__model = models.load_model(file_path)

class EpochEndCallback( callbacks.Callback ):

	def on_batch_end(self, batch, logs=None):
		super().on_epoch_end( batch , logs)
		winsound.PlaySound('C://beep.wav', winsound.SND_FILENAME)


