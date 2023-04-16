
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import numpy as np
import os

LABELS_WF = 'labels_for360_images'
images_path = 'cars_360images'
testFiles = 'output/test.txt'


class NN:
    def __init__(self):
        # Create a VGG16 model with weights pre-trained on ImageNet
        vgg = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(240, 240, 3)))
        vgg.trainable = False
        features = vgg.output
        features = Flatten()(features)
        l1 = Dense(128, activation="relu")(features)
        l2 = Dense(64, activation="relu")(l1)
        l3 = Dense(32, activation="relu")(l2)
        l4 = Dense(4, activation="sigmoid")(l3)
        self.cnn = Model(inputs=vgg.input, outputs=l4)

def train_model():
    # Load and preprocess data
    data = []
    targets = []
    filenames = []
    preProcess(data, filenames, targets)
    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")

    # Split data into training and testing sets
    split = train_test_split(data, targets, filenames, test_size=0.1, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainTargets, testTargets) = split[2:4]
    (trainFilenames, testFilenames) = split[4:]

    # Write test filenames to a file
    f = open(testFiles, "w")
    f.write("\n".join(testFilenames))
    f.close()

    # Train and save the model
    opt = Adam(lr=1e-3)
    model = NN().cnn
    model.compile(loss="mse", optimizer=opt)
    model.fit(trainImages, trainTargets,validation_data=(testImages, testTargets),epochs=25,verbose=1)
    model.save('model', save_format="h5")


def preprocess_row(row):
    filename, x_start, y_start, x_end, y_end = row.split(",")
    image_path = os.path.join(images_path, filename)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    x_start_norm = float(x_start) / width
    y_start_norm = float(y_start) / height
    x_end_norm = float(x_end) / width
    y_end_norm = float(y_end) / height
    resized_image = load_img(image_path, target_size=(240, 240))
    array_image = img_to_array(resized_image)
    return array_image, (x_start_norm, y_start_norm, x_end_norm, y_end_norm), filename

def preProcess(data, filenames, targets):
    rows = open(LABELS_WF).read().strip().split("\n")
    for row in rows:
        array_image, target, filename = preprocess_row(row)
        data.append(array_image)
        targets.append(target)
        filenames.append(filename)

if __name__ == '__main__':
    train_model()
