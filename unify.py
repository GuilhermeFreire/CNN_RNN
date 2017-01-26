import json
import os
import codecs

def list_files(dir='../drawception_dataset/DB'):
	r = []
	for root, dirs, files in os.walk(dir):
		for name in files:
			if(name[-5:] == ".json"):
				r.append(os.path.join(root, name))
	return r

def format(path, broken_images):
	with open(path) as json_data:
	    d = json.load(json_data)
	    text = ""
	    for example in d:
	    	img_path = example["img"]
	    	if(img_path not in broken_images):
	    		text += img_path + "||" + example["prevSent"] + "\n"
	    return text

def get_broken_images():
	path="/root/Documents/TESI/drawception_dataset/DB/"
	with open("broken_images.txt") as f:
		broken_images = f.readlines()
		for i in range(len(broken_images)):
			broken_images[i] = broken_images[i][len(path):-1]
	return broken_images


final_text = ""
brk_img = get_broken_images()
files = list_files()

for f in files:
	print(f)
	final_text += format(f, brk_img)

#print(final_text)
print("Done.")
with codecs.open("./metadata.txt", "w", encoding="utf8") as meta:
	meta.write(final_text)
print("File saved.")