from PIL import Image
import numpy as np

def flatten_images(image_paths):
    flattened_images = []
    for path in image_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img_array = np.array(img).flatten()  # Flatten the image to a 1D array
        flattened_images.append(img_array)
    return np.array(flattened_images)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x)

def 