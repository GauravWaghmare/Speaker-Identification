from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16 # Gaurav : Number of features 
timesteps = 8 # Gaurav : Number of states 
nb_classes = 10 # Gaurav : Number of users

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True, # Gaurav : 32 neurons in the first hidden layer
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # Gaurav : 32 neurons in the second hidden layer
model.add(LSTM(32))  # Gaurav : 32 neurons in the third hidden layer
model.add(Dense(10, activation='softmax')) # Gaurav : 10 neurons in output layer 

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim)) # Gaurav : Sample x state vector x feature
y_train = np.random.random((1000, nb_classes)) # Gaurav : Sample x user

# generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim)) # Gaurav : Sample x state vector x feature
y_val = np.random.random((100, nb_classes)) # Gaurav : Sample x user

model.fit(x_train, y_train,
          batch_size=64, nb_epoch=5,
          validation_data=(x_val, y_val))