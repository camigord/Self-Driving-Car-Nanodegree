import csv
import numpy as np
import pickle
import matplotlib.image as mpimg

# Folders containing training data
folders = ['./data', './data2']
lines = []
for folder in folders:
    with open(folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    image = mpimg.imread(source_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    # Augment data by flipping the images
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement

    images.append(image_flipped)
    measurements.append(measurement_flipped)

# Convert to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

# Save training data
pickle.dump((X_train, y_train), open( "train.p", "wb" ) )

# Load data
# X_train, y_train = pickle.load( open( "train.p", "rb" ) )
