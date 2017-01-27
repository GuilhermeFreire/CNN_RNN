import matplotlib.image as img
import numpy as np
import os

NB_IMAGES = 100

class helper:
    def __init__(self):
        self.listCharacter = []                         #HELP SCRIPT Lista de caracteres
        self.dictCharacter = {}                         #Dicionário é só usado no helper
        self.files_path = []                            #Caminho de cada arquivos
        self.encoded_Labels = []                        #HELP SCRIPT Labels com o indice de cada caracter
        self.maxLengthSentence = 0                      #HELP SCRIPT Tamanho da maior frase
        self.readMetaData()
        self.nb_characters = len(self.dictCharacter)    #HELP SCRIPT Quantidade de caracteres    
        self.images = []                                #HELP SCRIPT Matriz de Imagem
        self.readImage()
        self.npArrayEverything()
        self.fillEncoded_labes()
        self.nb_characters += 1
        self.listCharacter.append("<>")

    def fillEncoded_labes(self):
        maxLength = 0
        for item in self.encoded_Labels:
            if len(item) > maxLength:
                maxLength = len(item)
        self.maxLengthSentence = maxLength
        new_encoded_Labels = []
        for item in self.encoded_Labels:
            newItem = np.lib.pad(item, (0,maxLength-len(item)), "constant", constant_values=(self.maxLengthSentence))
            new_encoded_Labels.append(newItem)
        self.encoded_Labels = np.array(new_encoded_Labels)
        
    def npArrayEverything(self):
        self.images = np.array(self.images)
        self.encoded_Labels = np.array(self.encoded_Labels)

    def listDifferentChar(self, list_sentence):
        for sentence in list_sentence:
            for letter in sentence:
                if letter not in self.listCharacter:
                    self.listCharacter.append(letter)
        self.listCharacter.sort()
        self.dictCharacter = {k: v for v, k in enumerate(self.listCharacter)}

    def readMetaData(self):
        labels = []
        with open("metadata.txt", "r") as f:
            text = f.readlines()
       
        for item in text[:NB_IMAGES]:
            curr_item = item.split("||")
            self.files_path.append(curr_item[0])
            labels.append(curr_item[1].rstrip())
        
        self.listDifferentChar(labels)
        self.labelEncoder(labels)

    def labelEncoder(self, labels):
        for i, sentence in enumerate(labels):
            l_encoded_Labels = []                      
            for letter in sentence:              
                l_encoded_Labels.append(self.dictCharacter[letter])
            self.encoded_Labels.append(l_encoded_Labels)

    def readImage(self):
        for i, item in enumerate(self.files_path):          
            try:
                self.images.append(img.imread(item))
            except FileNotFoundError:
                pass


   #def readImages(files_path):
    #    X = []    
    #    for item in files_path:
            


    #image = img.imread("aeae.jpg")


#print(image.shape)
#print(image)


