import keras 
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Convolution2D, LSTM, Dense, RepeatVector, Embedding, MaxPooling2D, TimeDistributed, Activation, Merge
from keras.constraints import maxnorm
from keras.layers import Flatten
from keras.preprocessing.text import one_hot
import helper as hs

num_examples = 5

helper = hs.Helper()
helper.load_dataset()

#Data for train and test:
X_image, Y_labels = helper.load_n_random_examples(num_examples)
X_test_generator = helper.load_single_image(helper.img_paths[1])

curr_description = np.zeros((num_examples, helper.max_description_length))


#----------------------------#
#language_model:
language_model = Sequential()
language_model.add(Embedding(helper.num_char, 256, input_length=helper.max_description_length))
language_model.add(LSTM(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))

#image_model:
image_model = Sequential()

image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(142, 170, 3)))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Flatten())
image_model.add(Dense(128))

image_model.add(RepeatVector(helper.max_description_length))

#model:
model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))

model.add(LSTM(256, return_sequences=False))
model.add(Dense(helper.num_char))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train
print("Training...")

for i in range(helper.max_description_length):
    print(i)    
    next_char = np.array(Y_labels)[:,i]    
    onehot_next_char = helper.one_hot_vector(next_char)

    curr_description = np.roll(curr_description, -1, axis=1)
    curr_description[:, -1] = next_char

    model.fit([np.array(X_image), curr_description] , onehot_next_char, batch_size=32, nb_epoch=10, verbose=1)

#Save model:

#Predict:

for i in range(helper.max_description_length):   
    

    model.fit([np.array(X_image), curr_description] , onehot_next_char, batch_size=32, nb_epoch=10, verbose=1)


predictions = model.predict(np.array([X_test_generator]))
sentences = helper.reverse_one_hot(predictions)

print("")
print("")
print("SENTENCES:")    
print(sentences)
print("--")

gc.collect()

