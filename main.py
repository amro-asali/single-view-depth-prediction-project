from __future__ import print_function
import os
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def resize_image(image_path, x_start, y_end, y_start):
    """
        Resizes the image at the specified path and returns relevant information.

        Parameters:
        image_path (str): The path to the image to be resized.
        x_start (float): The x-coordinate at which the object in the image starts.
        y_end (float): The y-coordinate at which the object in the image ends.
        y_start (float): The y-coordinate at which the object in the image starts.

        Returns:
        tuple: A tuple containing the height of the resized image, the resized image, the x-coordinate at which the object
        starts, the y-coordinate at which the object ends, and the y-coordinate at which the object starts.
    """
    original_image = cv2.imread(image_path)
    resized_image = imutils.resize(original_image, width=240)
    (h, w) = resized_image.shape[:2]
    x_start = int(x_start * w)
    y_start = int(y_start * h)
    y_end = int(y_end * h)
    return h, resized_image, x_start, y_end, y_start


def load_image(image_filename):
    """
       Loads the image at the specified path, resizes it, converts it to an array, and returns it along with the path to the image.

       Parameters:
       image_filename (str): The name of the image file to be loaded.

       Returns:
       tuple: A tuple containing the image array and the path to the image.
    """
    image_path = os.path.sep.join(['cars_me', image_filename])
    image = load_img(image_path, target_size=(240, 240))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image_path


def display_result(image_filename, object_depth, resized_image, x_start, y_end, y_start):
    """
       Displays the result of the depth estimation on the specified image.

       Parameters:
       image_filename (str): The name of the image file.
       object_depth (float): The estimated distance to the object in the image.
       resized_image (np.ndarray): The resized image.
       x_start (float): The x-coordinate at which the object in the image starts.
       y_end (float): The y-coordinate at which the object in the image ends.
       y_start (float): The y-coordinate at which the object in the image starts.

       Returns:
       None
    """
    cv2.rectangle(resized_image, (x_start, y_start), (x_start, y_end), (0, 255, 0), 2)
    depth_text = f"Distance: {object_depth:.2f} "
    cv2.putText(resized_image, depth_text, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Output", resized_image)
    cv2.imwrite(f'{image_filename}', resized_image)
    cv2.waitKey(0)


def estimate_depth(real_world_height, focal_length, censor_height):
    """
        Loads the trained object detection model and applies it on a set of test images to estimate the depth of
        an object in the image.

        Args:
            real_world_height (float): The real world height of the object in meters.
            focal_length (float): The focal length of the camera in millimeters.
            censor_height (float): The height of the camera sensor in millimeters.

        Returns:
            None: The function saves the output image in the same directory as the input image with the estimated depth
            and bounding box drawn around the detected object.
    """
    model = load_model('model')
    test_images = ['5.jpg', '23.jpg', '25.jpeg', '28.jpeg']
    for image_filename in test_images:
        image_array, image_path = load_image(image_filename)
        (x_start, y_start, x_end, y_end) = model.predict(image_array)[0]
        image_height_pixels, resized_image, x_start, y_end, y_start = resize_image(image_path, x_start, y_end, y_start)
        object_height_pixels = y_end - y_start
        object_depth = (image_height_pixels * real_world_height * focal_length) / (object_height_pixels * censor_height)
        display_result(image_filename, object_depth, resized_image, x_start, y_end, y_start)


if __name__ == '__main__':
    # estimate depth for an suv with known real world height of 1.57 meters
    estimate_depth(real_world_height=1.57, focal_length=4.1, censor_height=5.22)
