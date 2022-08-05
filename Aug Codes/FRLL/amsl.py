# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:25:04 2021

@author: ahmed
"""
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )

dataset=[]
import numpy as np
from skimage import io
import os
from PIL import Image

image_directory = '/content/drive/MyDrive/FRLL/morph_amsl/'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if(image_name.split('.')[1]=='jpg'):
        image=io.imread(image_directory + image_name)
        image=Image.fromarray(image,'RGB')
        #image=image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        print(i)

x = np.array(dataset)

i = 0
for batch in datagen.flow(x, batch_size=16,
                          save_to_dir='/content/drive/MyDrive/FRLL/Augmentedmorph_amsl',
                          save_prefix='aug',
                          save_format='jpg'
                          ):
    i+=1
    if i > 271 :
        break
