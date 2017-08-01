import os
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.regularizers import l2

# Function to preprocess and to load data_folder
from utils.utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 15, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

def main(_):
    BATCH_SIZE = FLAGS.batch_size
    EPOCHS = FLAGS.epochs

    '''
    1st. Loading the training data
    '''
    # Folders containing provided samples and self-collected images
    data_folders = ['./data','./data2']

    image_paths, steering_angles = get_data_path(data_folders)

    # Split data into training and validation
    image_paths_train, image_paths_valid, steering_train, steering_valid = train_test_split(image_paths, steering_angles, test_size=0.2, random_state=15)

    # Training data generator: Augments the data by adding different distortions as shown in jupyter notebook
    train_generator = generator(image_paths_train, steering_train, batch_size=BATCH_SIZE)
    # Validation data generator: Does not distort the images
    validation_generator = generator(image_paths_valid, steering_valid, batch_size=BATCH_SIZE, validation_flag=True)

    model = Sequential()
    # Normalizing input data
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3)))
    # Cropping the input image to remove sky and front of the car
    model.add(Cropping2D(cropping=((50,25), (0,0))))
    # Convolutional layers
    model.add(Conv2D(filters=24,kernel_size=5,strides=2,activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(filters=36,kernel_size=5,strides=2,activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(filters=48,kernel_size=5,strides=2,activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='elu', kernel_regularizer=l2(0.001)))
    # Flattening the final conv layer output
    model.add(Flatten())
    model.add(Dropout(0.8))
    # Fully connected layers
    model.add(Dense(100, activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Dense(50, activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Dense(10, activation='elu', kernel_regularizer=l2(0.001)))
    #model.add(Dense(1, activation='tanh'))
    model.add(Dense(1))

    #model.add(MaxPooling2D(pool_size=2)

    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=optimizer)
    history_object = model.fit_generator(train_generator, steps_per_epoch= int(len(image_paths_train)/BATCH_SIZE), \
            validation_data=validation_generator, validation_steps=int(len(image_paths_valid)/BATCH_SIZE), epochs=EPOCHS)

    #print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save('model.h5')

if __name__ == '__main__':
    tf.app.run()
