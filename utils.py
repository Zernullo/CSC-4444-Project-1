# Author: Daniel Guo
# Reference: ChatGPT, Gemini
import os
import numpy as np
from PIL import Image

"""
Load MNIST images from folders named 0-9.
Returns: images, labels
The purpose of this file is to load images for the other scripts.
"""
#flatten is true in the parameter automatically because
def load_mnist(path, flatten=True, normalize=True): #Reads image files from folders 0-9 and return them as NumPy arrays with labels
    images, labels = [], []
    for digit in range(10): #loops through each folder named after digits 0-9
        digit_path = os.path.join(path, str(digit))
        if not os.path.exists(digit_path): #skip if folder does not exist
            continue
        for file in os.listdir(digit_path): #Read .png or .jpg images from each folder 
            if file.endswith('.png') or file.endswith('.jpg'):
                img = np.array(Image.open(os.path.join(digit_path, file))).astype(np.float32) #Convert image to NumPy array
                if normalize: #Convert pixel values to [0,1] range
                    img /= 255.0
                if flatten: #Flatten 2D image (28x28) to 1D vector length (784)
                    img = img.flatten()
                images.append(img)
                labels.append(digit)
    return np.array(images), np.array(labels) #Return images and labels as NumPy arrays because they are easier to work with in Machine Learning tasks
