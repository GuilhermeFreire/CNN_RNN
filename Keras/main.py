import keras 
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Convolution2D, LSTM, Dense, RepeatVector, Embedding, MaxPooling2D, TimeDistributed, Activation, Merge
from keras.constraints import maxnorm
from keras.layers import Flatten
from keras.preprocessing.text import one_hot
import helper as hs

num_examples = 1

helper = hs.Helper()
helper.load_dataset()

#Data for train and test:
X_image, Y_labels = helper.load_n_random_examples(num_examples)
X_test_image = helper.load_single_image(helper.img_paths[1])

curr_description = np.array([-1]*num_examples*helper.max_description_length).reshape(num_examples, helper.max_description_length)


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

    model.fit([np.array(X_image), curr_description] , onehot_next_char, batch_size=32, nb_epoch=1000, verbose=1)

#Save model:
#-----------------------------------------#
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
#------------------------------------------#

#Predict:
description = np.array([-1]*helper.max_description_length)
for i in range(helper.max_description_length):   
    curr_char = model.predict([np.array([X_test_image]), np.array([description])])
    curr_char = np.argmax(curr_char[0])
    description = np.roll(description, -1)    
    description[-1] = curr_char

description = helper.convert_vec_index_to_string(description)

print("")
print("")
print("SENTENCES:")    
print(description)
print("--")

gc.collect()

