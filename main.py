from __future__ import print_function

import os

import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def estimateDepth(real_world_height,focal_length,censor_height):
    # Load the trained model
    model = load_model('model')

    # Process each test image
    test_images = ['5.jpg', '23.jpg', '25.jpeg', '28.jpeg']
    for image_filename in test_images:
        # Load the image and preprocess it
        image_path = os.path.sep.join(['cars_me', image_filename])
        image = load_img(image_path, target_size=(240, 240))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Use the trained model to predict the bounding box coordinates
        (x_start, y_start, x_end, y_end) = model.predict(image_array)[0]

        # Resize the original image and convert the bounding box coordinates to pixel values
        original_image = cv2.imread(image_path)
        resized_image = imutils.resize(original_image, width=240)
        (h, w) = resized_image.shape[:2]
        x_start = int(x_start * w)
        y_start = int(y_start * h)
        y_end = int(y_end * h)

        # Calculate the depth of the object in the image
        object_height_pixels = y_end - y_start
        object_depth_cm = (h * real_world_height * focal_length) / (object_height_pixels * censor_height)

        # Draw the bounding box and depth value on the image, then display it
        cv2.rectangle(resized_image, (x_start, y_start), (x_start, y_end), (0, 255, 0), 2)
        depth_text = f"Distance: {object_depth_cm:.2f} "
        cv2.putText(resized_image, depth_text, (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Output", resized_image)
        cv2.imwrite(f'{image_filename}',resized_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    # estimate depth for an suv with known real world height of 1.57 meters
    estimateDepth(real_world_height=1.57,focal_length=4.1,censor_height=5.22)
