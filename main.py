#Alexander Breeze   101 143 291
#Michael Balcerzak  101 071 699
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
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

def drawRectangles(image,chords):
 image2=ImageDraw.Draw(image)
 for chord in chords:
  image2.rectangle(chord, fill=None, outline="red", width=3)
 return image

#takes 2 arrays of img, returns array of generated img and array of corresponding sprite img
def generateData(backgrounds,sprites,numInstances,minSpriteCount,maxSpriteCount,rotation,sizing,flipping):
 data=[] #array of compound images
 y=[] #array of sprite locations in each compound image
 for _ in range(numInstances):
  y.append([])
  sprite=random.choice(sprites)
  compound=random.choice(backgrounds).copy().resize((1024,1024))
  for _ in range(random.randint(minSpriteCount,maxSpriteCount)):
   tempSprite=sprite.copy()
   tempSprite.thumbnail((512,512),PIL.Image.LANCZOS)
   if rotation:
    tempSprite=tempSprite.rotate(random.randint(0,360), expand=1) #true makes it resize to fit new image. Uses nearest neighbor to keep pixel colors
   if sizing:
    newSize=random.randint(64,512)
    tempSprite.thumbnail((newSize,newSize),PIL.Image.NEAREST)
   if flipping and random.randint(0,1)==0:
    tempSprite=tempSprite.transpose(Image.FLIP_LEFT_RIGHT)
   spriteWidth,spriteHeight=tempSprite.size
   spriteX=random.randint(0,1024-spriteWidth)
   spriteY=random.randint(0,1024-spriteHeight)
   tempSprite.convert("RGBA")
   compound.paste(tempSprite, (spriteX,spriteY), tempSprite) #last argument is to apply transparent background
   y[-1].append([spriteX,spriteY,spriteX+spriteWidth,spriteY+spriteHeight])
  data.append(compound.getdata())
  compound=drawRectangles(compound,y[-1])
 data=np.array(data).reshape((numInstances,1024,1024,3))
 return data, y

#takes the locations of 2 folders of images, returns 2 numpy arrays of those images
def readData():
 backgrounds = []
 for x in range(1,80):
  backgrounds.append(Image.open("./backgrounds/background"+str(x)+".jpg"))
 sprites = []
 for x in range(1,101):
  sprites.append(Image.open("./sprites/sprite"+str(x)+".png"))
 return backgrounds, sprites
b,s=readData()
data,y=generateData(b,s,10,2,4,1,1,1)
#print(data)
print(len(data))
print(len(data[0]))
print(len(data[0][0]))
print(len(data[0][0][0]))
print(y)
