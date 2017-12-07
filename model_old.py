import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Lambda

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
	#lines.pop(0)

images = []
measurements = []
correction = 0.2

for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'data/IMG/'+filename
		image = cv2.imread(current_path)
		images.append(image)
	steering_center = float(line[3])
	steering_left = steering_center + correction
	steering_right = steering_center - correction
	measurements.append(steering_center)
	measurements.append(steering_left)
	measurements.append(steering_right)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(np.fliplr(image))
	augmented_measurements.append(-measurement)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

inpshp = (160, 320, 3)

model = Sequential()
#model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=inpshp))
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=inpshp))

model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=3)

model.save('model.h5')

