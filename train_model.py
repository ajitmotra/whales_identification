##############################  Import Some packages  ############################
import csv
import sys
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn import preprocessing
from keras.optimizers import SGD
from keras.optimizers import Adam
import hashlib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
import math
from skimage import color
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import model_from_json
import cv2
#####  ##### 


############### Keep some variables here
batch_size = 4
re_size = 160, 60
learning_rate = 0.0001
X_train = []
y_train=[]
X_val = []
y_val = []
img_rows = re_size[1]
img_cols = re_size[0]


#### Read Val and Test data ### 
#### Two laps of data saved in saparate folder so that it does not leak into main #### training set
with open("./simulator-linux/driving_log.csv") as csv_file:
	fileread = csv.reader(csv_file)
	for row in fileread:		
		image = Image.open(row[0])
		image.load()	
		image = image.crop((0,40, 320, 160))
		image.thumbnail(re_size, Image.ANTIALIAS)
		image1 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
		y_val.append(float(row[3]))
		X_val.append(np.array(image))		
		image.close()

X_val = np.array(X_val)
y_val = np.array(y_val, dtype=float)
print(X_val.shape)
print(y_val.shape)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

##################         Training data is huge and just not possible to read that ##################         at once so added a generator

def generate_arrays_from_file(path):
		while 1:
			csv_file = open("./driving_log.csv")
			fileread = csv.reader(csv_file)
			l = list(fileread)
			random.shuffle(l)
			for row in l:
				X_train = []
				y_train = []	
				#center	
				image = Image.open(row[0])
				image.load()	
				image = image.crop((0,40, 320, 160))
				image.thumbnail(re_size, Image.ANTIALIAS)
				image1 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
				image.close()
				#left
				image = Image.open(row[1])
				image.load()	
				image = image.crop((0,40, 320, 160))
				image.thumbnail(re_size, Image.ANTIALIAS)
				image2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
				image.close()
				#Right
				image = Image.open(row[2])
				image.load()	
				image = image.crop((0,40, 320, 160))
				image.thumbnail(re_size, Image.ANTIALIAS)
				image3 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
				image.close()
				#image1 = normalize_image(np.array(image1))  ## Now added in model it self
				#print(float(row[3]))                        ## Comment out to check frames read
				#plt.imshow(image1)							
				#plt.show()
				X_train.append(np.array(image1))  # center 	
				y_train.append(float(row[3]))					
				X_train.append(np.array(image2))  # Left 	
				y_train.append(float(row[3]) + 0.25)	
				X_train.append(np.array(image3))  # Right 	
				y_train.append(float(row[3]) - 0.25)					
				#X_train.append(np.fliplr(np.array(image1)))  # flipped
				#y_train.append(float(row[3])*-1.0) 				
				X_train = np.array(X_train)
				y_train = np.array(y_train, dtype=float)
				yield (X_train, y_train)				
				image.close()
			csv_file.close()


# number of convolutional filters to use
nb_filters1 = 3     # 
nb_filters2 = 16
nb_filters3 = 32
nb_filters4 = 64
nb_filters5 = 128

# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size1 = (3, 3)
kernel_size2 = (3, 3)
kernel_size3 = (3, 3)
kernel_size4 = (3, 3)
kernel_size5 = (1, 1)
### input shape
input_shape = (img_rows, img_cols, 3)

####
print(input_shape)
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Convolution2D(nb_filters1, kernel_size1[0], kernel_size1[1],
                        border_mode='valid',  init='normal'
                        ))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters2, kernel_size2[0], kernel_size2[1], init='normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters3, kernel_size3[0], kernel_size3[1], init='normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters4, kernel_size4[0], kernel_size4[1], init='normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters5, kernel_size5[0], kernel_size5[1], init='normal'))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, init='normal'))
model.add(Activation('relu'))

model.add(Dense(512, init='normal'))
model.add(Activation('relu'))


model.add(Dense(1000, init='normal'))
model.add(Activation('relu'))
model.add(Dense(3000, activation='softmax'))


//model.add(Dropout(0.5))

//model.add(Dense(1))

adam = Adam(lr=learning_rate)   ##### Also tried adam default but this works better for me

model.compile(loss='mean_squared_error',
              optimizer=adam, metrics=['accuracy'])

model.fit_generator(generate_arrays_from_file('./driving_log.csv'),
        samples_per_epoch=10000, nb_epoch=1, verbose=1, validation_data=(X_val,y_val))

score = model.evaluate(X_test, y_test, batch_size=4, verbose=1)
print('Test score:', score[0])

################### Save the Model Finally 
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")
print("Saved Model to the disk")
