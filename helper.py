import numpy as np
from PIL import Image

class Helper:
    def __init__(self):
        #Mude esse caminho para refletir o local do dataset
        self.complimentary_path = ""

        #Caracter especial End Of Sentence
        self.EOS = "<>"

        self.img_dims = (142,170,3)
        self.img_paths = []
        self.descriptions = []
        self.char_to_index = {}
        self.index_to_char = []
        self.num_examples = 0
        self.num_char = 0
        self.max_description_length = 0

    def separate_data(self, data):
        #Separa um dado no formato abaixo em uma tupla.
        #"caminho/para/img||Descrição dessa imagem"
        #("caminho/para/img", "Descrição dessa imagem")
        sep = data.split("||")
        return (sep[0], sep[1])

    def convert_char_to_index(self, char):
        return self.char_to_index[char]

    def convert_index_to_char(self, index):
        return self.index_to_char[index]

    def load_single_description(self, index):
        #Carrega uma descrição formatada com os índices dos caracteres
        text = self.descriptions[index]
        description = np.lib.pad(np.asarray(list(map(self.convert_char_to_index, text))), (0, self.max_description_length - len(text)), "constant", constant_values=self.char_to_index[self.EOS])
        return description

    def load_single_image(self, path):
        #Carrega uma imagem em um numpy array
        return np.asarray(Image.open(self.complimentary_path + path).getdata()).reshape(self.img_dims)

    def load_n_images(self, images):
        if(type(images) == int):
            examples = range(images)
        else:
            examples = images
        for i in examples:
            yield self.load_single_image(self.img_paths[i])

    def load_n_descriptions(self, descriptions):
        if(type(descriptions) == int):
            examples = range(descriptions)
        else:
            examples = descriptions
        for i in examples:
            yield self.load_single_description(i)


    def load_n_examples(self, n):
        #Carrega n exemplos sequenciais do dataset
        # for i in range(n):
            # yield (self.load_single_image(self.img_paths[i]), self.load_single_description(i))
        return (self.load_n_images(n), self.load_n_descriptions(n))

    def load_n_random_examples(self, n):
        #Carrega n exemplos aleatórios do dataset
        # for i in range(n):
        r = np.random.random_integers(0,self.num_examples, n)
        return list(self.load_n_images(r)), list(self.load_n_descriptions(r))
            # yield (self.load_single_image(self.img_paths[r]), self.load_single_description(r))

    def load_all_examples(self):
        return self.load_n_examples(self.num_examples)
        # for i in range(self.num_examples):
        #     yield (self.load_single_image(self.img_paths[i]), self.load_single_description(i))

    def load_dataset(self, meta_data_path="metadata.txt"):
        #Lê os metadados do arquivo
        with open(meta_data_path) as md:
            data = md.readlines()
            #Aplica a função separate_data em todas as linhas do arquivo
            #e separa em duas listas.
            self.img_paths, self.descriptions = list(zip(*map(self.separate_data, data)))

        #Guarda o número de exemplos do dataset
        self.num_examples = len(self.img_paths)

        #Guarda o tamanho máximo das descrições
        #Somamos 1, pois acrescentamos um "caracter" especial EOS ao final da frase.
        self.max_description_length = len(max(self.descriptions, key=len)) + 1

        #Gera o dicionário de conversão entre índice e caracter (ambos os sentidos)
        self.index_to_char = list(set("".join(map(str, self.descriptions))))
        self.index_to_char.sort()
        #Acrescentamos o "caracter" especial EOS
        self.index_to_char.append(self.EOS)
        self.char_to_index = {v:k for k,v in enumerate(self.index_to_char)}

        #Guarda o tamanho do espaço de caracteres
        self.num_char = len(self.index_to_char)

    def gen_epoch(self, batch_size, num_epochs):
        for i in range(num_epochs):
            yield self.load_n_random_examples(batch_size)

    def one_hot_char(self, index_char):
        one_hot = np.zeros(self.num_char)
        one_hot[index_char] = 1
        return one_hot

    def one_hot_vector(self, vector):
        return np.array(list(map(self.one_hot_char, vector)))

    def reverse_one_hot(self, description):
        sentences = []
        for sentence in predictions:
            sent = ""    
            for vec_char in sentence:
                nb = vec_char.argmax()
                sent += self.index_to_char[nb]
        sentences.append(sent)
        return sentences
  
    def convert_vec_index_to_string(self, description):
        return "".join(map(self.convert_index_to_char, description))

    #def one_hot(self, labels):
    #   return map(self.one_hot_vector, labels)

if __name__ == "__main__":
    h = Helper()
    h.load_dataset()

    # for X, Y in h.load_n_random_examples(5):
    #     print("X:", np.asarray(X).shape, " Y:", np.asarray(Y).shape)
    #     print(Y, end="\n\n")
    X, Y = h.load_n_random_examples(5)
    #print(X)
    #print(Y)

    #for i in X:
    #    print(i)

    #for i in Y:
    #    print(i)

    ran = np.random.random_integers(0,h.num_examples, 3)
    test = h.load_n_images(ran)
    #for t in test:
        #print(t)
    #print(ran)

    print("Total characters in dataset:", np.sum(list(map(lambda s: len(s), h.descriptions))))
