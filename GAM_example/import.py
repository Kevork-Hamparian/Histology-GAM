

#%%
import cv2
import os
import matplotlib as plt
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

import tensorflow as tf

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


images=load_images_from_folder(r'C:\Users\Kevork\Documents\Histology-GAM\BACH_images')
images=np.array(images)

plt.imshow(images[0])


# %%
