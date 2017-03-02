import numpy as np
from scipy import misc
from nltk import word_tokenize

class Helper():
	def __init__(self):
	#Mude esse caminho para refletir o local do dataset
		self.complementary_path = "../drawception_dataset/Resized_DB/"

		#Caracteres especiais para Start of Sentence e End Of Sentence
		self.SOS = "<SOS>"
		self.EOS = "<EOS>"

		self.img_dim = ()
		self.img_paths = []
		self.descriptions = []
		self.char_to_index = {}
		self.index_to_char = []
		self.num_examples = 0
		self.num_char = 0
		self.max_description_length = 0

	def load_dataset(self, meta_data_path="metadata.txt"):
		#Lê os metadados do arquivo
		with open(meta_data_path) as md:
			data = md.readlines()
			#Aplica a função separate_data em todas as linhas do arquivo
			#e separa em duas listas.
			self.img_paths, self.descriptions = list(zip(*map(lambda s: s.split("||"), data)))

		#Guarda o número de exemplos do dataset
		self.num_examples = len(self.img_paths)

		#Guarda o tamanho máximo das descrições
		#Somamos 2, pois acrescentamos os "caracteres" especiais SOS e EOS na frase.
		self.max_description_length = len(max(self.descriptions, key=len)) + 2

		#Gera o dicionário de conversão entre índice e caracter (ambos os sentidos)
		self.index_to_char = list(set("".join(self.descriptions)))
		self.index_to_char.sort()
		#Acrescentamos os "caracteres" especiais SOS e EOS
		self.index_to_char.extend([self.SOS, self.EOS])
		self.char_to_index = {v:k for k,v in enumerate(self.index_to_char)}

		#Guarda o tamanho do espaço de caracteres
		self.num_char = len(self.index_to_char)

		temp_img = self.get_images(0,1)[0]
		self.img_dim = temp_img.shape
		print("IMG SHAPE:", self.img_dim)

	def load_dataset_words(self, meta_data_path="metadata.txt"):
		#Lê os metadados do arquivo
		with open(meta_data_path) as md:
			data = md.readlines()
			#Aplica a função separate_data em todas as linhas do arquivo
			#e separa em duas listas.
			self.img_paths, self.descriptions = list(zip(*map(lambda s: s.split("||"), data)))

		#Guarda o número de exemplos do dataset
		self.num_examples = len(self.img_paths)

		self.descriptions = list(map(lambda x: word_tokenize(x.lower()), self.descriptions))

		#Guarda o tamanho máximo das descrições
		#Somamos 2, pois acrescentamos os "caracteres" especiais SOS e EOS na frase.
		self.max_description_length = len(max(self.descriptions, key=len)) + 2
		print(self.num_examples)

		#Gera o dicionário de conversão entre índice e caracter (ambos os sentidos)
		self.index_to_char = list(set([word for description in self.descriptions for word in description]))
		self.index_to_char.sort()
		#Acrescentamos os "caracteres" especiais SOS e EOS
		self.index_to_char.extend([self.SOS, self.EOS])
		self.char_to_index = {v:k for k,v in enumerate(self.index_to_char)}

		#Guarda o tamanho do espaço de caracteres
		self.num_char = len(self.index_to_char)

		temp_img = self.get_images(0,1)[0]
		self.img_dim = temp_img.shape
		print("IMG SHAPE:", self.img_dim)

	def pad(self, text):
		padded = [self.char_to_index[self.SOS]]
		padded.extend(text)
		padded.extend([self.char_to_index[self.EOS]]*(self.num_steps - len(padded)))
		return padded

	def conv_char_to_index(self, text):
		return list(map(lambda s: self.char_to_index[s], text))

	def get_images(self, start_index, size):
		select_img_paths = self.img_paths[start_index:start_index+size]
		imgs = []
		for path in select_img_paths:
			imgs.append(misc.imread(self.complementary_path + path))
		return np.array(imgs, dtype=np.uint8)

	def dataset_iterator(self, batch_size, num_steps):
		self.num_steps = num_steps + 1
		num_batches = self.num_examples // batch_size
		if num_batches == 0:
			raise ValueError("num_batches == 0, decrease batch_size or num_steps")

		converted_descriptions = map(self.conv_char_to_index, self.descriptions)
		padded_descriptions = np.array(list(map(self.pad, converted_descriptions)))

		for i in range(num_batches):
			#Tomar cuidado com o num_steps na hora do treinamento
			#Isso devolve num_steps-1
			#Acho que corrigi
			x = padded_descriptions[i*batch_size:(i+1)*batch_size,:-1]
			y = padded_descriptions[i*batch_size:(i+1)*batch_size,1:]
			img = self.get_images(i*batch_size,batch_size)
			yield(img, x, y)

	def load_test_dataset(self, meta_data_path="test_metadata.txt"):
		with open(meta_data_path, "r") as md:
			data = md.readlines()
			img_paths, descriptions = list(zip(*map(lambda s: s.split("||"), data)))
		
		imgs = []
		for path in img_paths:
			imgs.append(misc.imread(path))

		return (np.array(imgs), descriptions)




# h = Helper()
# h.load_dataset_words()
# print(len(h.index_to_char))