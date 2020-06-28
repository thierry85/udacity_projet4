import pickle
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
import pandas as pd

# Setup Keras
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# load information in CSV given in the project
# format: center,left,right,steering,throttle,brake,speed
simulation_data=pd.read_csv('./data/driving_log.csv',sep=',')

# build data for model
simulation_images_center=simulation_data["center"]
simulation_images_left=simulation_data["left"]
simulation_images_right=simulation_data["right"]
simulation_steering=simulation_data["steering"]


images=[]
angles=[]
for i in range(1,len(simulation_data)):
  image=mpimg.imread('./data/'+simulation_images_center[i])
  images.append(image)
  angles.append(simulation_steering[i])
  image=mpimg.imread('./data/IMG/'+simulation_images_left[i].split('/')[-1])
  images.append(image)
  # correct angle for left camera
  angles.append((float(simulation_steering[i])+0.2))
  image=mpimg.imread('./data/IMG/'+simulation_images_right[i].split('/')[-1])
  images.append(image)
  # correct angle for right camera
  angles.append((float(simulation_steering[i])-0.2))

yb_train =np.array(angles)
Xb_train=np.array(images)

#Flipping Images And Steering Measurements
Xf_train = np.fliplr(Xb_train)
yf_train = -yb_train

#increase dataset
X_train= np.concatenate((Xb_train,Xf_train), axis=0)
y_train= np.concatenate((yb_train,yf_train),axis=0)

#shuffle and split
X_train, y_train = shuffle(X_train, y_train)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=832289)


print("data generated")

#building the model
model = Sequential()

# normalisation of images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# set up cropping2D layer
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# model based on Nvidia Dropout layers have been added to avoid overfitting

#convolution part
model.add(Conv2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Conv2D(64,3,3, activation="relu"))
model.add(Conv2D(64,3,3, activation="relu"))


model.add(Flatten())


# fully connected layers
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
# output the sterring
model.add(Dense(1))

print("model build")

# generate and run the models
model.compile(loss='mse',optimizer='adam')

#model parameters
BATCH_SIZE = 100
EPOCHS = 15

model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=2, validation_data=(X_valid, y_valid))

#saving
model.save('model.h5')
print("weights saved")

# print the model
model.summary()

