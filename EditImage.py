"""
Crops a certain 16:9 area of an image, scales it to 1280x720, and adds it to the bottom of the image
 - This means that the new effective resolution is 1280x1440
 - Run this individually for both wet and dry, and change the SAVEPATH variable accordingly
"""

import os
import numpy as np
import cv2 as cv2


BASEPATH = "***DRY/WET PATH***"

with os.scandir(BASEPATH) as entries:
    for entry in entries:
        if entry.is_file():
            print(entry.name)
            image = cv2.imread(BASEPATH + "/" + entry.name)
            cropped = image[406:575, 490:790].copy()
            resized_cropped = cv2.resize(cropped, (1280, 720)) 
            vis = np.concatenate((image, resized_cropped), axis=0)

            OUTPUTPATH = "***OUTPUT PATH***"
            cv2.imwrite(os.path.join(OUTPUTPATH, entry.name), vis)