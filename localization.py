# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:38:58 2018

@author: Administrator
"""

from skimage.io import imread
from skimage.io import imsave
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import cv2 
import skimage
from skimage.util import *
str=input('Enter file name :')
car_image = imread(str, as_gray=True)

# it should be a 2 dimensional array
print(car_image.shape)

# the next line is not compulsory however, a grey scale pixel
# in skimage ranges between 0 & 1. multiplying it with 255
# will make it range between 0 & 255 (something we can relate better with

gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
skimage.io.imsave("binary.jpg", img_as_uint(binary_car_image))
plt.show()