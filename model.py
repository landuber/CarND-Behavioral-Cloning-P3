
# coding: utf-8

# In[86]:


import csv
import os.path

root_dir = './data'
#root_dir = './data_old'
csv_path = os.path.join(root_dir, 'driving_log.csv')
imgs_dir = os.path.join(root_dir, 'IMG/')
samples = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
print(len(samples))
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.3)
      
      
import cv2
import numpy as np
import sklearn
import random


def load_data(samples):
    images = []
    measurements = []
    for sample in samples:
        steering_center = float(sample[3])

        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        getImage = lambda filename: cv2.imread(imgs_dir + filename.split('/')[-1])

        images.extend((getImage(sample[0]), getImage(sample[1]), getImage(sample[2])))
        measurements.extend((steering_center, steering_left, steering_right))


    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    return X_train, y_train
    

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                getImage = lambda filename: cv2.imread(imgs_dir + filename.split('/')[-1])

                images.extend((getImage(batch_sample[0]), getImage(batch_sample[1]), getImage(batch_sample[2])))
                measurements.extend((steering_center, steering_left, steering_right))


            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            
            

train_generator = generator(train_samples, batch_size=256)
validation_generator = generator(validation_samples, batch_size=256)

X_train, y_train = load_data(samples[1:])


# In[87]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
#from keras.layers.core import Dropout
from keras.optimizers import Adam

model = Sequential()
#model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(BatchNormalization())
model.add(Conv2D(24,(5,5), kernel_regularizer=regularizers.l2(0.001), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(36,(5,5), kernel_regularizer=regularizers.l2(0.001), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(48,(5,5), kernel_regularizer=regularizers.l2(0.001), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(64,(3,3), kernel_regularizer=regularizers.l2(0.001), padding='valid', activation='relu', strides=(1,1)))
model.add(Conv2D(64,(3,3), kernel_regularizer=regularizers.l2(0.001), padding='valid', activation='relu', strides=(1,1)))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=1e-4))
history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=6)
print(model.summary())

#history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)*6,
#                                     validation_data=validation_generator, 
#                                     validation_steps=len(validation_samples)*6, max_queue_size=1000000, use_multiprocessing=True, epochs=3)
model.save('model.h5')


# In[29]:


import matplotlib.pyplot as plt

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[ ]:




