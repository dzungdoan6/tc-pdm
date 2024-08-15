import os, glob, cv2
from PIL import Image
from pathlib import Path



def resize_to_320x256(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Resize the image to 320x256
    resized_image = image.resize((320, 256))
    return resized_image

def resize_to_320x256_crop_to_256x256(image_path):
    # Resize the image to 320x256
    resized_image = resize_to_320x256(image_path)

    # Calculate the cropping box coordinates for center crop
    left = (320 - 256) // 2  # Calculate the left coordinate for center crop
    top = (256 - 256) // 2  # Calculate the top coordinate for center crop
    right = left + 256
    bottom = top + 256

    # Crop the image from the center
    cropped_image = resized_image.crop((left, top, right, bottom))
    # cropped_image.show()  # Show the cropped image
    return cropped_image

    
    


