import keras 
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM

X_image = np.array([[[0,0],[0,0],[0,0],[0,0]]])
Y_character = np.array([[[0],[1],[1],[0]]])

X_test = np.array([[[1,0],[1,0],[1,0],[1,0]]])

model = Sequential()
model.add(LSTM(1, input_shape=(4, 2), return_sequences=True))
model.add(LSTM(1, input_shape=(4, 1), return_sequences=True))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train
model.fit(X_image, Y_character)

#Predict
predictions = model.predict(X_test)

print(predictions)

import gc; gc.collect()
