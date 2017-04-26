# model.py
# Author : Thomas Tartiere

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# correction factor for left and right camera
USE_SIDE_CAMERA = False
CORRECTION = 0.15

# Load images and steering measurements
samples = []
with open('training/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# build training and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create batches of samples to reduce memory size
def generator(samples,batch_size=128):

	num_samples = len(samples)
	while True:
		shuffle(samples)  # shuffle the samples

		for offset in range(0,num_samples,batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			steering_angles = []
			for line in batch_samples:
				path = ""

				# load images
				img_center = cv2.imread(path+line[0])
				img_left = cv2.imread(path+line[1])
				img_right = cv2.imread(path+line[2])

				if img_center is not None:

					# calculate steering for left and right images
					steering_center = float(line[3])
					steering_left = steering_center+CORRECTION
					steering_right = steering_center-CORRECTION

					# add the center image to the data set
					images.append(img_center)
					steering_angles.append(steering_center)

					if USE_SIDE_CAMERA:

						if img_left is not None:
							# add the left image to the data set
							images.append(img_left)
							steering_angles.append(steering_left)

						if img_right is not None:
							# add the right image to the data set
							images.append(img_right)
							steering_angles.append(steering_right)

					# add flipped image to the dataset
					image_flipped = np.fliplr(img_center)
					measurement_flipped = -steering_center
					images.append(image_flipped)
					steering_angles.append(measurement_flipped)


			X_train = np.array(images)
			y_train = np.array(steering_angles)
			yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# -------------------------------------
# Deep learning model
# -------------------------------------

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse',optimizer="adam")
model.fit_generator(train_generator,samples_per_epoch=len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples),epochs=2)

# save model
model.save('model.h5')