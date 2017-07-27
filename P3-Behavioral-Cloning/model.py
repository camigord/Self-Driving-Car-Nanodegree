import os
import csv
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

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_folder', './data3', "Bottleneck features training file (.p)")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = FLAGS.data_folder + '/IMG/'+batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(source_path)
                center_angle = float(batch_sample[3])

                # Randomly flipping the image to augment data
                if np.random.random_sample() >= 0.5:
                    center_image = np.fliplr(center_image)
                    center_angle = -center_angle

                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def main(_):
    BATCH_SIZE = FLAGS.batch_size
    samples = []
    with open(FLAGS.data_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    # Remove head line
    samples = samples[1:]

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    model = Sequential()
    # Normalizing input data
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,25), (0,0))))
    model.add(Conv2D(filters=24,kernel_size=5,strides=2,activation='elu'))
    model.add(Conv2D(filters=36,kernel_size=5,strides=2,activation='elu'))
    model.add(Conv2D(filters=48,kernel_size=5,strides=2,activation='elu'))
    model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='elu'))
    model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.8))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.8))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.8))
    #model.add(Dense(1, activation='tanh'))
    model.add(Dense(1))

    #model.add(MaxPooling2D(pool_size=2)

    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=optimizer)
    history_object = model.fit_generator(train_generator, steps_per_epoch= int(len(train_samples)/BATCH_SIZE), \
            validation_data=validation_generator, validation_steps=int(len(validation_samples)/BATCH_SIZE), epochs=FLAGS.epochs)

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
