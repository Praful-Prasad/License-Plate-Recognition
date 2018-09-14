# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:43:11 2018

@author: Administrator
"""

from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization
import cv2
from skimage import util
import numpy as np
from skimage.io import imsave
# this gets all the connected regions and groups them together
label_image = measure.label(localization.binary_car_image)

# getting the maximum width, height and minimum width and height that a license plate can be
plate_dimensions = (0.05*label_image.shape[0], 0.1*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []
fig, (ax1) = plt.subplots(1)
ax1.imshow(localization.gray_car_image, cmap="gray");

# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    if region.area < 50:
        #if the region is so small then it's likely not a license plate
        continue

    # the bounding box coordinates
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    # ensuring that the region identified satisfies the condition of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        plate_like_objects.append(localization.binary_car_image[min_row:max_row,
                                  min_col:max_col])
        crop_img = localization.car_image[min_row:max_row,min_col:max_col]
        imsave("plate.jpg",crop_img)
        plate_objects_cordinates.append((min_row, min_col,
                                              max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
    # let's draw a red rectangle over those regions

#Saving image in binary -
img = cv2.imread('plate.jpg',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imwrite('BW_plate.jpg',thresh1)

#Erosion - 
img = cv2.imread('BW_plate.jpg',0)
cv2.imshow('Original Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
kernel = np.ones((20,20),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
inverted_img = util.invert(img)

cv2.imwrite('inverted_BW_plate.jpg',inverted_img)
cv2.imshow('inverted image',inverted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(inverted_img,kernel,iterations = 1)
cv2.imshow('eroded inverted image',erosion)
cv2.imwrite("Eroded_inverted_image.jpg",erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()



plt.show()                                