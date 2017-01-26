import matplotlib.image as img
import numpy as np
import os

#print(os.listdir())

class helper:
    def __init__(self):
        self.listCharacter = []
        self.dictCharacter = {}
        self.encoded_Labels = []
        self.readMetaData()
        

    def listDifferentChar(self, list_sentence):
        for sentence in list_sentence:
            for letter in sentence:
                if letter not in self.listCharacter:
                    self.listCharacter.append(letter)
        self.listCharacter.sort()
        self.dictCharacter = {k: v for v, k in enumerate(self.listCharacter)}

    def readMetaData(self):
        files_path = []
        labels = []
        with open("metadata.txt", "r") as f:
            text = f.readlines()
       
        for item in text[:2]:
            curr_item = item.split("||")
            files_path.append(curr_item[0])
            labels.append(curr_item[1].rstrip())
        
        self.listDifferentChar(labels)
        self.labelEncoder(labels)

        return files_path, labels 

    def labelEncoder(self, labels):
        l_encoded_Labels = []
        for sentence in labels:
            for letter in sentence:
                l_encoded_Labels.append(self.dictCharacter[letter])
            self.encoded_Labels.append(l_encoded_Labels)
        print(self.encoded_Labels)
        print(self.dictCharacter)

h = helper()


   #def readImages(files_path):
    #    X = []    
    #    for item in files_path:
            


    #image = img.imread("aeae.jpg")


#print(image.shape)
#print(image)


