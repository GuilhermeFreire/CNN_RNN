import keras 
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Convolution2D, LSTM, Dense, RepeatVector
from keras.constraints import maxnorm
from keras.layers import Flatten

X_image = np.array([[[[0,2,1,0],[0,1,2,0]]]])
Y_image = np.array([[[1],[1],[1],[1]]])


Y_character = np.array([[[0],[1],[1],[0]]])


#X_test = np.array([[[1,0],[1,0],[1,0],[1,0]]])
X_test = np.array([[[[0,2,1,0],[0,1,2,0]]]])

model = Sequential()

model.add(Convolution2D(5, 2, 2, input_shape=(1, 2, 4), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
#model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(3, activation='tanh'))

model.add(RepeatVector(4))
model.add(LSTM(1, input_shape=(4, 3), return_sequences=True))
model.add(LSTM(1, return_sequences=True))




model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train
model.fit(X_image, Y_image)

#Predict
predictions = model.predict(X_test)

print(predictions)

gc.collect()

