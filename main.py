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
def generateData(backgrounds,sprites,numInstances,minSpriteCount,maxSpriteCount,rotation,sizing,flipping):
 data=np.array([]) #array of compound images
 y=np.array([]) #array of sprite locations in each compound image
 for _ in range(numInstances):
  y.append([])
  sprite=random.choice(sprites).copy().thumbnail((512,512),PIL.Image.ANTIALIAS).convert("RGBA")
  compound=random.choice(backgrounds).copy().resize((1024,1024))
  for _ in range(random.randint(minSpriteCount,maxSpriteCount)):
   if rotation:
    tempSprite=sprite.rotate(random.randint(0,360),true) #true makes it resize to fit new image. Uses nearest neighbor to keep pixel colors
   if sizing:
    newSize=random.randint(32,512)
    tempSprite=tempSprite.template((newSize,newSize),PIL.Image.NEAREST)
   if flipping and random.randion(0,1)==0:
    sprite.transpose(FLIP_LEFT_RIGHT)
   spriteWidth,spriteHeight=tempSprite.size()
   spriteX=random.randint(0,1024-width)
   spriteY=random.randint(0,1024-height)
   compound.paste(tempSprite, (spriteY,spriteX), tempSprite) #last argument is to apply transparent background
   y[-1].append([spriteY,spriteX,spriteY+height,spriteX+width])
   PIL.ImageDraw.Draw.rectangle([spriteY,spriteX,spriteY+height,spriteX+width], fill=None, outline="red")
  data.append(compound)
  compound.show()
  exit()
 return data, y

#takes the locations of 2 folders of images, returns 2 numpy arrays of those images
def readData():
 backgrounds = []
 for x in range(1,81):
  backgrounds.append(Image.open("./backgrounds/background"+str(x)+".jpg"))
 sprites = []
 for x in range(1,101):
  sprites.append(Image.open("./sprites/background"+str(x)+".jpg"))
 backgrounds[0].show()
 files = glob.glob ("./backgrounds/*.jpg")
 for myFile in files:
  image = cv2.imread (myFile)
  backgrounds.append (image)
 sprites = []
 files = glob.glob ("./sprites/*.png")
 for myFile in files:
  image = cv2.imread (myFile)
  sprites.append (image)
 backgrounds[0].show()
 sprites[0].show()
 return backgrounds, sprites
#b,s=readData()
#generateData(b,s,10,2,4,0,0,0)

def cleanData():
 exit() #this served its purpose, don't run it again
 currIndex=1
 for x in range(1,100):
  try:
   image=Image.open("./badbackgrounds/background"+str(x)+".jpg").convert('RGB')
   image.save("./backgrounds/background"+str(currIndex)+".jpg")
   currIndex+=1
  except:
   try:
    image=Image.open("./badbackgrounds/background"+str(x)+".JPG").convert('RGB')
    image.save("./backgrounds/background"+str(currIndex)+".jpg")
    currIndex+=1
   except:
    print("a")