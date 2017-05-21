import numpy as np
import tensorflow as tf
from helper import Helper
import codecs, json

h = Helper()
h.load_dataset_words()

#Generator de exemplos para treinar a rede
def gen_epochs(n, num_steps, batch_size):
	for i in range(n):
		yield h.dataset_iterator(batch_size, num_steps)

#Reseta o grafo computacional do TensorFlow
def reset_graph():
	if("sess" in globals() and sess):
		sess.close()
	tf.reset_default_graph()

#Treina a rede
def train_network(g, num_epochs, num_steps=68, batch_size=32, save=False):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		total_losses = []
		for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
			loss = 0
			steps = 0
			current_state = None
			for X_images, X, Y in epoch:
				steps += 1
				img_vector = sess.run([g["img_vector"]], feed_dict={g["img"]: X_images})
				feed_dict = {g['x']: X, g['y']: Y, g['init_state']: img_vector}
				current_loss, current_state, _ = sess.run([g["total_loss"], g["final_state"], g["train_step"]], feed_dict)
				loss += current_loss

			print("Average training loss for Epoch", idx, ":", loss/steps)
			total_losses.append(loss/steps)
			# if(isinstance(save, str)):
			# 	print("Saving at Epoch", idx, ":", loss/steps)
			# 	g["saver"].save(sess, save)

		if(isinstance(save, str)):
			g["saver"].save(sess, save)

	return total_losses

def _variable_with_weight_decay(name, shape, stddev, wd):
	#Evita redeclarações de variáveis do tensorflow quando chamando uma função múltiplas vezes
	variable = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(variable), wd, name="weight_loss")
		tf.add_to_collection("losses", weight_decay)
	return variable

def _variable_on_cpu(name, shape, initializer):
	#Evita redeclarações de variáveis do tensorflow quando chamando uma função múltiplas vezes
	with tf.device("/cpu:0"):
		variable = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	return variable

def image_representation(images, batch_size, state_size, training=False):
	if(training):
		distorted_images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
		distorted_images = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=63), distorted_images)
		distorted_images = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), distorted_images)
		distorted_images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), distorted_images)
	else:
		distorted_images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)


	#Camada 1 de convolução
	with tf.variable_scope("conv1") as scope:
		kernel = _variable_with_weight_decay("weights", shape=[5,5,3,64], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv2d(distorted_images, kernel, [1,1,1,1], padding="SAME")
		biases = _variable_on_cpu("biases", [64], tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope.name)

	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="pool")
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="norm1")

	#Camada 2 de convolução
	with tf.variable_scope("conv2") as scope:
		kernel = _variable_with_weight_decay("weights", shape=[5,5,64,64], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding="SAME")
		biases = _variable_on_cpu("biases", [64], tf.constant_initializer(0.1))
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope.name)
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="norm2")
	pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="pool2")

	with tf.variable_scope("fully1") as scope:
		reshaped = tf.reshape(pool2, [batch_size, -1])
		dim = reshaped.get_shape()[1].value
		weights = _variable_with_weight_decay("weights", shape=[dim, 384], stddev=0.04, wd=0.004)
		biases = _variable_on_cpu("biases", [384], tf.constant_initializer(0.1))
		fully1 = tf.nn.relu(tf.matmul(reshaped, weights) + biases, name=scope.name)

	with tf.variable_scope("fully2") as scope:
		weights = _variable_with_weight_decay("weights", [384, state_size], stddev=0.04, wd=0.004)
		biases = _variable_on_cpu("biases", [state_size], tf.constant_initializer(0.1))
		fully2 = tf.nn.relu(tf.matmul(fully1, weights) + biases, name=scope.name)

	return fully2

def gen_rnn_cell(cell_type, state_size):
    if(cell_type == "GRU"):
        rnn_cell = tf.contrib.rnn.GRUCell(state_size)
    else:
        rnn_cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
    return rnn_cell

def build_graph(cell_type=None, state_size=100, batch_size=32, num_steps=200, num_classes=h.num_char, num_layers=3, learning_rate=5e-4, training=False):
	reset_graph()

	x = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_placeholder")
	y = tf.placeholder(tf.int32, [batch_size, num_steps], name="labels_placeholder")

	img_tensor_dims = [batch_size]
	img_tensor_dims.extend(h.img_dim)
	img = tf.placeholder(tf.float32, img_tensor_dims, name="images_placeholder")

	img_vector = image_representation(img, batch_size, state_size, training)

	embeddings = tf.get_variable("embedding_matrix", [num_classes, state_size])

	rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

	if(cell_type == "LSTM"):
		cell = tf.contrib.rnn.MultiRNNCell([gen_rnn_cell(cell_type, state_size) for _ in range(num_layers)], state_is_tuple=True)
	else:
		cell = tf.contrib.rnn.MultiRNNCell([gen_rnn_cell(cell_type, state_size) for _ in range(num_layers)])

	init_state = cell.zero_state(batch_size, tf.float32)

	img_vector = tuple([img_vector] * num_layers)

	rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
	
	with tf.variable_scope("softmax"):
		W = tf.get_variable("W", [state_size, num_classes])
		b = tf.get_variable("b", [num_classes], initializer=tf.constant_initializer(0.0))

	rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
	y_reshaped = tf.reshape(y, [-1])

	logits = tf.matmul(rnn_outputs, W) + b

	predictions = tf.nn.softmax(logits)

	total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

	return dict(x = x, y = y, init_state = init_state, img = img, img_vector = img_vector, final_state = final_state, total_loss = total_loss, train_step = train_step, preds = predictions, saver = tf.train.Saver(tf.global_variables()))

def gen_description(g, checkpoint, num_words, img_src, starting_word=h.SOS, top_n_words=None):

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		g["saver"].restore(sess, checkpoint)

		img_vector = sess.run([g["img_vector"]], feed_dict={g["img"]: [img_src]})

		state = None
		current_word = h.char_to_index[starting_word]
		words = [current_word]

		saved_states = []

		for i in range(num_words):
			if state is not None:
				feed_dict={g["x"]: [[current_word]], g["init_state"]: state}
			else:
				feed_dict={g["x"]: [[current_word]], g["init_state"]: img_vector}

			preds, state = sess.run([g["preds"], g["final_state"]], feed_dict)

			saved_states.append(state)

			if top_n_words is not None:
				p = np.squeeze(preds)
				p[np.argsort(p)[:-top_n_words]] = 0
				p = p / np.sum(p)
				current_word = np.random.choice(h.num_char, 1, p=p)[0]
			else:
				current_word = np.random.choice(h.num_char, 1, p=np.squeeze(preds))[0]

			words.append(current_word)
	words = list(map(lambda x: h.index_to_char[x], words))
	
	saved_states = np.squeeze(np.array(saved_states))
	saved_states_json = saved_states.tolist()
	json.dump(saved_states_json, codecs.open("LSTM_states_words.json", "w", encoding="utf-8"), separators =(",", ":"), sort_keys=True)
	return words
	
def choose_n_unique(num, variation, prob):
	words=[]
	while len(words) < variation:
		temp = np.random.choice(num, 1, p=prob)[0]
		if(h.index_to_char[temp] == h.EOS):
			words.append(temp)
			continue;
		if(temp not in words):
			words.append(temp)
			prob[temp] = 0
			prob = prob / np.sum(prob)
	return words
	
def beam_search(g, checkpoint, num_words, img_src, starting_word=h.SOS, top_n_sentences=10, variation=10):

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		g["saver"].restore(sess, checkpoint)

		img_vector = sess.run([g["img_vector"]], feed_dict={g["img"]: [img_src]})

		current_words = [h.char_to_index[starting_word]]*top_n_sentences
		sentences = [([h.char_to_index[starting_word]],1.0)]*top_n_sentences

		saved_states = [img_vector]*top_n_sentences

		for i in range(num_words):
			for sent_idx in range(top_n_sentences):
				feed_dict={g["x"]: [[sentences[sent_idx][0][-1]]], g["init_state"]: saved_states[sent_idx]}

				preds, state = sess.run([g["preds"], g["final_state"]], feed_dict)

				saved_states[sent_idx] = state

				#topN = sorted(np.squeeze(preds), reverse=True)[:top_n_senteces]

				p = np.squeeze(preds)
				p[np.argsort(p)[:-top_n_sentences]] = 0
				p = p / np.sum(p)
				next_words = choose_n_unique(h.num_char, variation, p)
				prob_words = [(next_word, p[next_word]) for next_word in next_words]
					
				current_sent = sentences[sent_idx]
				
				for next_word in prob_words:
					sentences.append((current_sent[0] + [next_word[0]], current_sent[1] * next_word[1]))
						
			sentences = sorted(sentences[10:], key=lambda x: x[1], reverse=True)[:top_n_sentences]


	word_sentences = [list(map(lambda x: h.index_to_char[x], sent[0])) for sent in sentences]
	
#	saved_states = np.squeeze(np.array(saved_states))
#	saved_states_json = saved_states.tolist()
#	json.dump(saved_states_json, codecs.open("LSTM_states_words.json", "w", encoding="utf-8"), separators =(",", ":"), sort_keys=True)
	return word_sentences

#g = build_graph(cell_type="GRU", num_steps=h.max_description_length, state_size=512, batch_size=32, training=True)
#losses = train_network(g, 100, num_steps=h.max_description_length, batch_size=32, save="saves/drawception/Word_Model_100_epochs_512_state_retraining")

#t = h.load_test_dataset()
#g = build_graph(cell_type="GRU", num_steps=1, batch_size=1, state_size=512)
#for i in range(len(t[0])):
#	desc = gen_description(g, "saves/drawception/Word_Model_100_epochs_512_state_retraining", 68, img_src = t[0][0], starting_word=h.SOS, top_n_words=5)
#formated_sentence = ""
#for word in desc:
#	if(word[0] == "'"):
#		formated_sentence += word
#	else:
#		formated_sentence += " " + word
#print(formated_sentence)

t = h.load_test_dataset()
g = build_graph(cell_type="GRU", num_steps=1, batch_size=1, state_size=512)

with open("possible_responses.txt", "w", encoding="utf-8") as f:
	for j in range(len(t[0])):
		final_sentences = []
		f.write("================================="+str(j+1)+"=================================\n")
		f.write("Frase: "+ t[1][j])
		for i in range(20):
			desc = gen_description(g, "saves/drawception/Word_Model_100_epochs_512_state_retraining", 68, img_src = t[0][j], starting_word=h.SOS, top_n_words=5)
			formated_sentence = ""
			for word in desc:
				if(word[0] == "'"):
					formated_sentence += word
				else:
					formated_sentence += " " + word
			f.write(str(i+1)+": "+formated_sentence+"\n")
		f.write("====================================================================\n")
		f.flush()

#t = h.load_test_dataset()
#g = build_graph(cell_type="GRU", num_steps=1, batch_size=1, state_size=512)
#final_sentences = []
#for i in range(len(t[0])):
#	desc = gen_description(g, "saves/drawception/Word_Model_100_epochs_512_state_retraining", 68, img_src = t[0][i], starting_word=h.SOS, top_n_words=5)
#	formated_sentence = ""
#	for word in desc:
#		if(word[0] == "'"):
#			formated_sentence += word
#		else:
#			formated_sentence += " " + word
#	final_sentences.append(str(i+1)+": "+formated_sentence)
#
#for i in range(len(final_sentences)):
#	print(final_sentences[i])
#	print("========================================")
	
	
#descs = beam_search(g, "saves/drawception/Word_Model_100_epochs_512_state_retraining", 68, img_src = t[0][4], starting_word=h.SOS, top_n_sentences=10, variation=10)

#print(descs)
#final_sentences = []
#
#for desc in descs:
#	formated_sentence = ""
#	for word in desc:
#		if(word[0] == "'"):
#			formated_sentence += word
#		else:
#			formated_sentence += " " + word
#	final_sentences.append(formated_sentence.replace(" <EOS>", "").replace("<SOS> ", ""))
#print(final_sentences)
