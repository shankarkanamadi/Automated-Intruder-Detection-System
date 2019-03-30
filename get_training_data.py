import os
import cv2
import numpy as np
from keras.utils import to_categorical
def get_training_data(dir_path):
	dirs=os.listdir(dir_path)
	persons=[]
	for dir in dirs:
		dir_check=dir_path+"/"+dir
		if not os.path.isfile(dir_check):
			persons.append(dir)
		#print(y_train)

	x_train=[]
	y_train=[]

	person_dict=dict(zip(persons,np.arange(len(persons))))

	for person in persons:
		img_path=dir_path+"/"+person
		images=os.listdir(img_path)
	
		for image in images:
			img=cv2.imread(img_path+"/"+image)
			resized=cv2.resize(img,(140,140))
			x_train.append(resized)
			y_train.append(person_dict[person])
	
	x_train=np.array(x_train)
	y_train=np.array(y_train)

	x_train = x_train.reshape(-1, 140, 140, 3).astype('float32') / 255.
	y_train = to_categorical(y_train)

	print("Getting training data is done..")
	return x_train,y_train

#print(get_training_data("./data"))