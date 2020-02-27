import app.perceptron as ml
from PIL import Image
import os
import numpy as np

def load_images(path_to_folder):
    images = []
    for file in os.listdir(path_to_folder):
        if '.png' in file:
            img = np.array(Image.open(path_to_folder+"/"+file))
            img = img.flatten()/255
            images.append(img[0:12288])
    return np.array(images)

def normalize_data(images_array):
    images_mean = np.mean(images_array,axis=0)
    normalized_images = images_array - images_mean
    return normalized_images