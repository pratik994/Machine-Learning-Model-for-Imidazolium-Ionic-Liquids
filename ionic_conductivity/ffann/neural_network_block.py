from keras.wrappers.scikit_learn import KerasRegressor
from keras.activations import *
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import load_model
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model


config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)



def neural_network_block(dimension_length, lr,hidden_layer1_neuron,hidden_layer2_neuron,activation,X_train,y_train,epoch,batch_size):
    opt= Adam(learning_rate=lr)
    regressor = Sequential()
    regressor.add(Dense(units=hidden_layer1_neuron,activation=activation,input_dim=dimension_length)) ###number of first hidden nodes and input dimension
    regressor.add(Dense(units=hidden_layer2_neuron,activation=activation))  ## hidden second layer
    regressor.add(Dense(units=1,activation=activation))
    regressor.compile(optimizer= opt,loss='mse',metrics=['mae','acc','mse'])
    history=regressor.fit(X_train,y_train,batch_size =batch_size,epochs=epoch)




