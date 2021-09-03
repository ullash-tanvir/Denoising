import os
import time
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, redirect, url_for, request, render_template, app
from werkzeug.utils import secure_filename
import os
from flask import send_file
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

def preprocess_image(image_path):

  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("static/"+filename+".jpg")
    print("Saved as " +filename+".jpg")


def get_patches(file_name, patch_size, crop_sizes):
    '''This functions creates and return patches of given image with a specified patch_size'''
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    patches = []
    for crop_size in crop_sizes:  # We will crop the image to different sizes
        crop_h, crop_w = int(height * crop_size), int(width * crop_size)
        image_scaled = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h - patch_size + 1, patch_size):
            for j in range(0, crop_w - patch_size + 1, patch_size):
                x = image_scaled[i:i + patch_size,
                    j:j + patch_size]  # This gets the patch from the original image with size patch_size x patch_size
                patches.append(x)
    print("Get Patch Exit")
    return patches


def create_image_from_patches(patches, image_shape):
    '''This function takes the patches of images and reconstructs the image'''
    image = np.zeros(image_shape)  # Create a image with all zeros with desired image shape
    patch_size = patches.shape[1]
    p = 0
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):

        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            image[i:i + patch_size, j:j + patch_size] = patches[p]  # Assigning values of pixels from patches to image
            p += 1
    print("Create Image From Patch Exit")
    return np.array(image)


def predict_fun(model, image_path):
    # Creating patches for test image
    patches = get_patches(image_path, 40, [1])

    test_image = cv2.imread(image_path)

    patches = np.array(patches)
    ground_truth = create_image_from_patches(patches, test_image.shape)

    # predicting the output on the patches of test image

    patches = patches.astype('float32') / 255.
    patches_noisy = tf.clip_by_value(patches, clip_value_min=0., clip_value_max=1.)
    denoised_patches = model.predict(patches_noisy)
    denoised_patches = tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

    # Creating entire denoised image from denoised patches
    denoised_image = create_image_from_patches(denoised_patches, test_image.shape)
    print("Predcit Fun Exit")

    return denoised_image



# Define a flask app
application = Flask(__name__)
app=application
@app.route('/', methods=['GET'])

def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])

def upload():


    if request.method == 'POST':
        directory = "./static"
        files_in_directory = os.listdir(directory)
        filtered_files = [file for file in files_in_directory if file.endswith(".jpg") or file.endswith(".png")]
        for file in filtered_files:
            path_to_file = os.path.join(directory, file)
            os.remove(path_to_file)
        directory = "./uploads"
        files_in_directory = os.listdir(directory)
        filtered_files = [file for file in files_in_directory if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
        for file in filtered_files:
            path_to_file = os.path.join(directory, file)
            os.remove(path_to_file)

        SAVED_MODEL_PATH = r"esrgan-tf2_1"
        print("enter model download")
        model=hub.load(SAVED_MODEL_PATH)
        print("exit model download")

        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        hr_image = preprocess_image(file_path)
        print("HR Image")
        # img = image.load_img(file_path, target_size=(224, 224))
        fake_image = model(hr_image)
        fake_image = tf.squeeze(fake_image)




        save_image(tf.squeeze(fake_image), filename="Super_Resolution")
    return "Super_Resolution.jpg"
#
@app.route("/getimage")
def get_img():
    return "Super_Resolution.jpg"

@app.route('/noise', methods=['GET', 'POST'])
def denoise_image():

    if request.method == 'POST':
        directory = "./static"
        files_in_directory = os.listdir(directory)
        filtered_files = [file for file in files_in_directory if file.endswith(".jpg") or file.endswith(".png")]
        for file in filtered_files:
            path_to_file = os.path.join(directory, file)
            os.remove(path_to_file)

        directory = "./uploads"
        files_in_directory = os.listdir(directory)
        filtered_files = [file for file in files_in_directory if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
        for file in filtered_files:
            path_to_file = os.path.join(directory, file)
            os.remove(path_to_file)

        ridnet = tf.keras.models.load_model(r'ridnet.h5')

        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        denoised_image = predict_fun(ridnet, file_path)
        print("Image is to form from an array")
        image = Image.fromarray(tf.cast(denoised_image * 255, tf.uint8).numpy())
        image.save('static/Denoised_Image.jpg')
    return "Denoised_Image.jpg"


@app.route("/denoisedimage")
def get_denoisedimg():
    return "Denoised_Image.jpg"



if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=8080)
    app.run()