import keras 
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Convolution2D, LSTM, Dense, RepeatVector
from keras.constraints import maxnorm
from keras.layers import Flatten
from keras.preprocessing.text import one_hot
import helper as hs

database_size = 1024
n_steps_learning = 200

helper = hs.Helper()
helper.load_dataset()

#Data for train and test:
X_image_labels = helper.gen_epoch(batch_size = database_size, num_epochs = n_steps_learning)
X_test_generator = helper.load_single_image(helper.img_paths[1])

#Model:
model = Sequential()

model.add(Convolution2D(20, 5, 5, input_shape=(142, 170, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
#model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(384, activation='tanh'))
model.add(RepeatVector(helper.max_description_length))

model.add(LSTM(192, input_shape=(helper.max_description_length, 384), return_sequences=True))
model.add(LSTM(helper.num_char, return_sequences=True, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train
print("Training...")

i = 0  
for x,y in X_image_labels:
    print(i)
    i += 1    
    model.fit(np.array(x), np.array(y), batch_size=32, nb_epoch=10)
    
    predictions = model.predict(np.array([X_test_generator]))
    sentences = helper.reverse_one_hot(predictions)
    print("")
    print("")
    print("SENTENCES:")    
    print(sentences)
    print("--")

gc.collect()

