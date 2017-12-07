import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Lambda
from sklearn.model_selection import train_test_split
from random import shuffle

plt.switch_backend('agg')

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
	#lines.pop(0)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images, angles = [], []
			augmented_images, augmented_angles = [], []
			correction = 0.2
			for batch_sample in batch_samples:
				for i in range(3):
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					current_path = 'data/IMG/'+filename
					image = cv2.imread(current_path)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					images.append(image)
				steering_center = float(batch_sample[3])
				steering_left = steering_center + correction
				steering_right = steering_center - correction
				angles.append(steering_center)
				angles.append(steering_left)
				angles.append(steering_right)

			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				augmented_images.append(np.fliplr(image))
				augmented_angles.append(-angle)
			# trim image to only see section with road
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

batch_size=32
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

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
#model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=3)
history_obj = model.fit_generator(train_generator, samples_per_epoch= (3*len(train_samples)//batch_size)*batch_size, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=7)
print(history_obj.history.keys())

fig=plt.figure()
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show(block=True)
fig.savefig('test.png')

model.save('model.h5')

