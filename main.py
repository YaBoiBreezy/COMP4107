#Alexander Breeze   101 143 291
#Michael Balcerzak  101 071 699
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import glob
import random

# A function that implements a keras model with the sequential API
def sequential_model():
  model=keras.Sequential()
  model.add(keras.layers.Dense(12, input_shape=(4,), activation="relu"))
  model.add(keras.layers.Dense(6, activation="relu"))
  model.add(keras.layers.Dense(3, activation="softmax"))
  print(model.summary())
  optimizer=tf.keras.optimizers.SGD() #learning_rate=0.001
  model.compile(optimizer=optimizer, loss="cross_entropy")

  # A keras model
  return model

#takes 2 arrays of img, returns array of generated img and array of corresponding sprite img
def generateData(backgrounds,sprites,num_instances,minSpriteCount,maxSpriteCount,rotation,sizing):
 data=np.array([]) #array of compound images
 y=np.array([]) #array of sprite in each compound image
 for _ in range(num_instances):
  background=random.choice(backgrounds)
  sprite=random.choice(sprites)
  compound=background
  for _ in range(random.randint(minSpriteCount,maxSpriteCount)):
   if rotation:
    tempSprite=sprite.rotate(random.randint(0,360),true) #true makes it resize to fit new image. Uses nearest neighbor to keep pixel colors
   if sizing:
    tempSprite=tempSprite.resize(int(((random.random()*2)+0.5))*tempSprite.size,PIL.Image.NEAREST)
   background.paste(tempSprite, (random.randint(0,background.size[0]),random.randint(0,background.size[1])))

  data.append(compound)
  y.append(sprite)
  #data[0].save("./saveddatas.img")
 return data, y

#takes the locations of 2 folders of images, returns 2 numpy arrays of those images
def readData(backgroundsLocation,spritesLocation):
 backgrounds = np.array([])
 files = glob.glob (backgroundsLocation+"/*.img")
 for myFile in files:
  image = cv2.imread (myFile)
  backgrounds.append (image)
 print('X_data shape:', np.array(backgrounds).shape)
 sprites = np.array([])
 files = glob.glob (backgroundsLocation+"/*.img")
 for myFile in files:
  image = cv2.imread (myFile)
  sprites.append (image)
 print('X_data shape:', np.array(sprites).shape)
 return backgrounds, sprites
readData("./backgrounds/","./sprites/")
 