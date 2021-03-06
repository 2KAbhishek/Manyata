import re
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

IMG_DIR = '../images/'

# Grayscale
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise Removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

# Thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Opening - Erosion + Dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Deskew
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Example Image
image = cv2.imread(IMG_DIR + 'example.jpg')
b,g,r = cv2.split(image)
rgb_img = cv2.merge([r,g,b])

# Fix Rotation
#osd = pytesseract.image_to_osd(image)
#angle = re.search('(?<=Rotate: )\d+', osd).group(0)

#if angle == "90":
    #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#if angle == "180":
    #image = cv2.rotate(image, cv2.ROTATE_180)
#if angle == "270":
    #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Preprocess image
gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)

images = {'gray': gray,
          'thresh': thresh,
          'opening': opening
          }

#Output using Pytesseract
custom_config = r'-l hin'
# custom_config = r'-l hin -c tessedit_char_whitelist=0123456789 --psm 6'
print('Original Image')
print('-----------------------------------------')
print(pytesseract.image_to_string(image, config=custom_config))
print('\n-----------------------------------------')
print('Threshholded Image')
print('-----------------------------------------')
print(pytesseract.image_to_string(thresh, config=custom_config))
print('\n-----------------------------------------')
print('Opened Image')
print('-----------------------------------------')
print(pytesseract.image_to_string(opening, config=custom_config))
