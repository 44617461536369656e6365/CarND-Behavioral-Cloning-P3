import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('D:\\Development\\Udacity\\SDC\\windows_sim\\my_data\\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

# If we weren't running on a local machine (as in if this were AWS we 
# would need to update the path)
for line in lines:
    # Load the left, right and centre images
    for i in range(3):
        #print(line)
        source_path = line[i]
        tokens = source_path.split('\\')
        filename = tokens[-1]
        #local_path = "./data/" + filename
        local_path = "D:\\Development\\Udacity\\SDC\\windows_sim\\my_data\\IMG\\" + filename
        #print(local_path)
        image = cv2.imread(local_path)
        #print(image.shape)
        images.append(image)
    # Use for the left/right turn with a correction value
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
    #exit()



# Flip a copy of the training data to add right turns
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

print(len(images))
print(len(measurements))

#X_train = np.array(images)
#y_train = np.array(measurements)
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print(X_train.shape)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

#print(len(images))
#print(len(measurements))
#exit()
