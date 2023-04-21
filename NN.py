import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

LABELS_WF = 'labels_for360_images'
DATASET_IMAGES_PATH = 'cars_360images'
TEST_FILE_NAMES = 'output/test.txt'


class NN:
    """
        A neural network class that defines the architecture of the model used to predict the bounding boxes of cars
        in images.

        Attributes:
            cnn (keras.models.Model): The compiled Keras model that represents the neural network.
    """
    def __init__(self):
        # Create a VGG16 model with weights pre-trained on ImageNet
        vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(240, 240, 3)))
        vgg.trainable = False
        features = vgg.output
        features = Flatten()(features)
        l1 = Dense(128, activation="relu")(features)
        l2 = Dense(64, activation="relu")(l1)
        l3 = Dense(32, activation="relu")(l2)
        l4 = Dense(4, activation="sigmoid")(l3)
        self.cnn = Model(inputs=vgg.input, outputs=l4)


def preprocess_row(row):
    """
        Preprocesses a row from the labels file and returns the corresponding image array, target values, and filename.

        Args:
            row (str): A row from the labels file in the format "filename,x_start,y_start,x_end,y_end".

        Returns:
            tuple: A tuple of the image array, target values, and filename.
    """
    filename, x_start, y_start, x_end, y_end = row.split(",")
    image_path = os.path.join(DATASET_IMAGES_PATH, filename)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    x_start_norm = float(x_start) / width
    y_start_norm = float(y_start) / height
    x_end_norm = float(x_end) / width
    y_end_norm = float(y_end) / height
    resized_image = load_img(image_path, target_size=(240, 240))
    array_image = img_to_array(resized_image)
    return array_image, (x_start_norm, y_start_norm, x_end_norm, y_end_norm), filename


def pre_process(data, filenames, targets):
    """
    Reads the labels file, preprocesses each row, and appends the resulting image array, target values, and filename to
    the corresponding lists.

    Args:
        data (list): A list to which the image arrays will be appended.
        filenames (list): A list to which the filenames will be appended.
        targets (list): A list to which the target values will be appended.
    """
    rows = open(LABELS_WF).read().strip().split("\n")
    for row in rows:
        array_image, target, filename = preprocess_row(row)
        data.append(array_image)
        targets.append(target)
        filenames.append(filename)


def load_preprocessed_data():
    """
        Loads the preprocessed image data and target values from disk, normalizes the image data, and returns the result.

        Returns:
            tuple: A tuple of the preprocessed image data, filenames, and target values.
    """
    data = []
    targets = []
    filenames = []
    pre_process(data, filenames, targets)
    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")
    return data, filenames, targets


def split_data(data, filenames, targets):
    """
           Splits the image data, filenames, and target values into training and testing sets.

           Args:
               data (numpy.ndarray): The preprocessed image data.
               filenames (list): The filenames corresponding to each image in the data array.
               targets (numpy.ndarray): The target values corresponding to each image in the data array.

           Returns:
               tuple: A tuple of the testing filenames, testing image data, testing target values, training image data,
            and training target values.
    """
    split = train_test_split(data, targets, filenames, test_size=0.1, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainTargets, testTargets) = split[2:4]
    (trainFilenames, testFilenames) = split[4:]
    return testFilenames, testImages, testTargets, trainImages, trainTargets


def train_model():
    """
       Loads the preprocessed data, splits it into training and testing sets, trains a neural network model on the training
       data, and saves the resulting model to disk.
    """
    data, filenames, targets = load_preprocessed_data()
    # Split data into training and testing sets
    test_filenames, test_images, test_targets, train_images, train_targets = split_data(data, filenames, targets)
    # Write test filenames to a file
    f = open(TEST_FILE_NAMES, "w")
    f.write("\n".join(test_filenames))
    f.close()
    # Train and save the model
    opt = Adam(lr=1e-3)
    model = NN().cnn
    model.compile(loss="mse", optimizer=opt)
    model.fit(train_images, train_targets, validation_data=(test_images, test_targets), epochs=25, verbose=1)
    model.save('model', save_format="h5")


if __name__ == '__main__':
    train_model()
