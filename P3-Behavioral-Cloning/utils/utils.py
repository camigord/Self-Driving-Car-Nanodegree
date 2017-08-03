from sklearn.utils import shuffle
import numpy as np
import cv2
import csv

def get_data_path(data_folders, minimum_speed=10.0, angle_correction=0.25, target_avg_factor = 1.0, num_bins=40):
    '''
    This function gets the path to the training samples, adjust the angles for left/right images and discard those samples
    which were taken traveling at less than the minimum speed.
    The function also tries to control the distribution of training samples by analyzing the histogram of training angles and removing those bins which are
    over-represented.
    '''
    image_paths = []
    steering_angles = []

    # Get path to training images and angles
    for data_folder in data_folders:
        with open(data_folder + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if line[0] == 'center':
                    # Ignore header
                    continue
                else:
                    # Remove those samples which were taken when car wasn't moving too fast
                    if float(line[6]) > minimum_speed:
                        # Center image path and angle
                        source_path = data_folder + '/IMG/'+line[0].split('/')[-1]
                        image_paths.append(source_path)
                        steering_angles.append(float(line[3]))
                        # Left image path and angle
                        source_path = data_folder + '/IMG/'+line[1].split('/')[-1]
                        image_paths.append(source_path)
                        steering_angles.append(float(line[3])+angle_correction)
                        # Right image path and angle
                        source_path = data_folder + '/IMG/'+line[2].split('/')[-1]
                        image_paths.append(source_path)
                        steering_angles.append(float(line[3])-angle_correction)

    image_paths = np.array(image_paths)
    steering_angles = np.array(steering_angles)

    # Try to normalize the distribution of training samples as shown in jupyter notebook (data_preprocessing.ipynb)
    hist, bins = np.histogram(steering_angles, num_bins)
    avg_samples_per_bin = np.mean(hist)

    # Computing keep probability for each sample. For each bin, we try to keep samples proportionally to how over or under-represented is each 'categorie'.
    new_target_avg = avg_samples_per_bin * target_avg_factor
    keep_probs = []
    for i in range(num_bins):
        if hist[i] < new_target_avg:
            keep_probs.append(1.)
        else:
            keep_probs.append(1./(hist[i]/new_target_avg))

    # Remove samples according to probability of each bin
    idx_to_remove = []
    for i in range(len(steering_angles)):
        for j in range(num_bins):
            if steering_angles[i] >= bins[j] and steering_angles[i] <= bins[j+1]:
                # Delete with probability 1-keep_prob
                if np.random.random_sample() > keep_probs[j]:
                    idx_to_remove.append(i)

    image_paths = np.delete(image_paths, idx_to_remove, axis=0)
    steering_angles = np.delete(steering_angles, idx_to_remove)

    return image_paths, steering_angles


def preprocess_image(img):
    '''
    Adds gaussian blur and transforms BGR to YUV.
    '''
    new_img = cv2.GaussianBlur(img, (3,3), 0)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

def random_distortion(img):
    '''
    Adds random distortion to training dataset: random brightness, shadows and a random vertical shift
    of the horizon position
    '''
    new_img = img.astype(float)

    # Add random brightness
    value = np.random.randint(-28, 28)
    new_img[:,:,0] = np.minimum(np.maximum(new_img[:,:,0],0),255)

    # Add random shadow covering the entire height but random width
    img_height, img_width = new_img.shape[0:2]
    middle_point = np.random.randint(0,img_width)
    darkening = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:middle_point,0] *= darkening
    else:
        new_img[:,middle_point:img_width,0] *= darkening

    # Applying a perspective transform at the beginning of the horizon line
    horizon = 2*img_height/5    # Assumes horizon to be located at 2/5 of image height
    v_shift = np.random.randint(-img_height/8,img_height/8)   # Shifting horizon by up to 1/8

    # First points correspond to a rectangle surrounding the image below the horizon line
    pts1 = np.float32([[0,horizon],[img_width,horizon],[0,img_height],[img_width,img_height]])
    # Second set of points correspond to same rectangle plus a random vertical shift
    pts2 = np.float32([[0,horizon+v_shift],[img_width,horizon+v_shift],[0,img_height],[img_width,img_height]])

    # Getting the perspective transformation
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # pplying the perspective transformation
    new_img = cv2.warpPerspective(new_img,M,(img_width,img_height), borderMode=cv2.BORDER_REPLICATE)
    return new_img.astype(np.uint8)

def generator(image_paths, steering_angles, batch_size=32, validation_flag=False):
    '''
    Training batches generator. Does not distort the images if "validation_flag" is set to True
    '''
    num_samples = len(image_paths)
    while 1:
        image_paths, steering_angles = shuffle(image_paths, steering_angles)
        for offset in range(0, num_samples, batch_size):
            batch_images = image_paths[offset:offset+batch_size]
            batch_angles = steering_angles[offset:offset+batch_size]

            images = []
            angles = []

            for batch_angle ,batch_image in zip(batch_angles,batch_images):
                img = cv2.imread(batch_image)
                img = preprocess_image(img)

                if not validation_flag:
                    img = random_distortion(img)

                # Randomly flipping the image to augment data
                # Only augmenting rare examples (angle > ~0.3)
                if abs(batch_angle) > 0.3 and np.random.random_sample() >= 0.5:
                    img = cv2.flip(img, 1)
                    batch_angle *= -1

                images.append(img)
                angles.append(batch_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
