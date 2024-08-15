from PIL import Image, ImageFilter
import numpy as np
import cv2

def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv_to_pil(image):
    color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return Image.fromarray(color_coverted) 

def gaussian_edge_detection(image):
    image = pil_to_cv(image)
    image = image - cv2.GaussianBlur(image, (21, 21), 3)+127
    image = cv_to_pil(image)
    return image
