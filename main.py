import keras 
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Convolution2D, LSTM, Dense, RepeatVector
from keras.constraints import maxnorm
from keras.layers import Flatten
import helper as hs

NB_IMAGES = 500

helper = hs.Helper()
X_image = helper.images[:NB_IMAGES-1]
Y_char = helper.encoded_Labels[:NB_IMAGES-1]
X_test = helper.images[NB_IMAGES-1:]
print(len(helper.encoded_Labels[0]))
print(helper.maxLengthSentence)
#Y_char to onehot
one_hot_Y_char = []
for sentence in Y_char:
    newSentence = []
    for char in sentence:
        curr_hot = np.zeros(helper.nb_characters)
        curr_hot[char] = 1
        newSentence.append(curr_hot)
    one_hot_Y_char.append(newSentence)
one_hot_Y_char = np.array(one_hot_Y_char)
print(one_hot_Y_char.shape)
#one_hot_Y_char = one_hot_Y_char[:,:4,:]


#X_image = np.array([[[[0,2,1,0],[0,1,2,0]]]])
#Y_char = np.array([[[1],[1],[1],[1]]])


#X_test = np.array([[[1,0],[1,0],[1,0],[1,0]]])

model = Sequential()

model.add(Convolution2D(20, 3, 3, input_shape=X_image.shape[1:], border_mode='same', activation='relu', W_constraint=maxnorm(3)))
#model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(384, activation='tanh'))
model.add(RepeatVector(helper.maxLengthSentence))

model.add(LSTM(192, input_shape=(helper.maxLengthSentence, 384), return_sequences=True))
model.add(LSTM(helper.nb_characters, return_sequences=True, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train
model.fit(X_image, one_hot_Y_char)

#Predict
predictions = model.predict(X_test)
print(predictions)
sentences = []
for sentence in predictions:
    sent = ""    
    for vec_char in sentence:
        nb = vec_char.argmax()
        sent += helper.listCharacter[nb]
    sentences.append(sent)

print(sentences)

gc.collect()

