
from tensorflow.python.keras.callbacks import TensorBoard
from Model import Classifier
import numpy as np
import time

data_dimension = 28
num_channels = 3

X = np.load( 'processed_data/x.npy')
Y = np.load( 'processed_data/y.npy')
test_X = np.load( 'processed_data/test_x.npy')
test_Y = np.load( 'processed_data/test_y.npy')

print( X.shape )
print( Y.shape )
print( test_X.shape )
print( test_Y.shape )

X = X.reshape( ( X.shape[0] , data_dimension**2 * num_channels  ) ).astype( np.float32 )
test_X = test_X.reshape( ( test_X.shape[0] , data_dimension**2 * num_channels  ) ).astype( np.float32 )

classifier = Classifier( number_of_classes=10 )
classifier.load_model( 'models/model.h5')

parameters = {
    'batch_size' : 120 ,
    'epochs' : 3 ,
    'callbacks' : None , #[ TensorBoard( log_dir='logs/{}'.format( time.time() ) ) ] ,
    'val_data' : ( test_X , test_Y )
}

#classifier.fit( X , Y  , hyperparameters=parameters )
#classifier.save_model( 'models/model.h5')

loss , accuracy = classifier.evaluate( test_X , test_Y )
print( "Loss of {}".format( loss ) , "Accuracy of {} %".format( accuracy * 100 ) )



