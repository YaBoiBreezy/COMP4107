#Alexander Breeze   101 143 291
#Michael Balcerzak  101 071 699
#Ifeanyichukwu Obi  101 126 269

```
sources:
https://stackoverflow.com/questions/62756658/loss-function-for-yolo
https://machinelearningspace.com/yolov3-tensorflow-2-part-1/
```

import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras, linalg, map_fn
from tensorflow.keras import layers
from keras.models import Sequential
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
import cv2
import glob
import random

gridSize=16 #divide image into gridSize x gridSize quadrants
quadSize=1024/gridSize #number of pixels in each quadrant
#labels y=[dataPoint][xQuadrant][yQuadrant][confidence,xOffset,yOffset,width,height]

def custom_loss(y,yhat): #using lanbda function to have access to yhat while iterating over y
 c1=y #loss is minimum of sum of squares of manhattan distances between pair of points yhat and all pairs of points in y
 c2=yhat
 distances=map_fn(lambda c1: tf.add(tf.square(tf.add(tf.abs(tf.subtract(c1[0],c2[0])),tf.abs(tf.subtract(c1[1],c2[1])))),tf.square(tf.add(tf.abs(tf.subtract(c1[2],c2[2])),tf.abs(tf.subtract(c1[3],c2[3]))))))
 return tf.minimum(distances)

# A function that implements a keras model with the sequential API
def createModel(xTrain, yTrain, xVal, yVal):
 print("creating model")
 model = Sequential()
 model.add(layers.Conv2D(512, kernel_size=3, activation='sigmoid', input_shape=(1024, 1024, 3)))
 model.add(layers.MaxPooling2D(3))
 model.add(layers.Conv2D(512, kernel_size=5, activation='sigmoid'))
 model.add(layers.MaxPooling2D(3))
 model.add(layers.Conv2D(512, kernel_size=5, activation='sigmoid'))
 model.add(layers.MaxPooling2D(3))
 model.add(layers.Conv2D(512, kernel_size=5, activation='sigmoid'))
 model.add(layers.MaxPooling2D(3))
 model.add(layers.Conv2D(512, kernel_size=5, activation='sigmoid'))
 model.add(layers.MaxPooling2D(3))
 model.add(layers.Flatten())
 model.add(layers.Dense(2048, activation='sigmoid'))
 model.add(layers.Dense(2048, activation='sigmoid'))
 model.add(layers.Dense(1024, activation='sigmoid'))
 model.add(layers.Dense(512, activation='sigmoid'))
 model.add(layers.Dense(512, activation='sigmoid'))
 model.add(layers.Dense(4, activation='softplus'))

 model.compile(loss=custom_loss, optimizer='adam')
 out = model.fit(x=xTrain, y=yTrain, validation_data=[xVal, yVal], epochs=100)
 print(out)
 print(model)
 return out,model

def drawRectangles(image,label):
 image2=ImageDraw.Draw(image)
 for bigX in range(gridSize):
  for bigY in range(gridSize):
   l=label[bigX][bigY]
   if l[0]:
    x=int(l[1]+bigX*quadSize) #middle of sprite
    y=int(l[2]+bigY*quadSize)
    w=int(l[3]/2) #halfWidth, halfHeight
    h=int(l[4]/2)
   image2.rectangle([x-w,y-h,x+w,x+h], fill=None, outline="red", width=3)
 return image

#takes 2 arrays of img, returns array of generated img and array of corresponding sprite img
def generateData(backgrounds,sprites,numInstances,minSpriteCount,maxSpriteCount,rotation,sizing,flipping):
 data=[] #array of compound images
 y=[] #array of sprite locations in each compound image
 for _ in range(numInstances):
  y.append([[[0,0,0,0,0]]*gridSize]*gridSize)
  sprite=random.choice(sprites)
  compound=random.choice(backgrounds).copy()
  for _ in range(random.randint(minSpriteCount,maxSpriteCount)):
   tempSprite=sprite.copy()
   if rotation:
    tempSprite=tempSprite.rotate(random.randint(0,360), expand=1) #true makes it resize to fit new image. Uses nearest neighbor to keep pixel colors
   if sizing:
    newSize=random.randint(64,256)
    tempSprite.thumbnail((newSize,newSize),PIL.Image.NEAREST)
   if flipping and random.randint(0,1)==0:
    tempSprite=tempSprite.transpose(Image.FLIP_LEFT_RIGHT)
   spriteWidth,spriteHeight=tempSprite.size
   spriteX=random.randint(0,1024-spriteWidth)
   spriteY=random.randint(0,1024-spriteHeight)
   midX=int(spriteX+(spriteWidth/2))
   midY=int(spriteX+(spriteWidth/2))
   bigX=int(midX / quadSize)
   bigX=int(midY / quadSize)
   xOffset=int(midX % quadSize)
   yOffset=int(midY % quadSize)
   tempSprite.convert("RGBA")
   compound.paste(tempSprite, (spriteX,spriteY), tempSprite) #last argument is to apply transparent background
   y[-1][bigX][bigY]=[1,xOffset,yOffset,spriteWidth,spriteHeight] #may overwrite previous sprite, but unlikely so ignore
  data.append(compound.getdata())
  compound=drawRectangles(compound,y[-1])
 data=np.array(data).reshape((numInstances,1024,1024,3))
 y=np.array(y)
 return data, y

#takes the locations of 2 folders of images, returns 2 numpy arrays of those images
def readData():
 backgrounds = []
 for x in range(1,80):
  backgrounds.append(Image.open("./backgrounds/background"+str(x)+".jpg").resize((1024,1024)))
 sprites = []
 for x in range(100):
  img=Image.open("./sprites/sprite"+str(x)+".png")
  img.thumbnail((512,512),PIL.Image.LANCZOS)
  sprites.append(img)
 return backgrounds, sprites

def main():
 trainSize=8
 testSize=2
 backgrounds,sprites=readData()
 print("read data")
 b1, b2 = train_test_split(backgrounds, test_size=0.25) #split backgrounds such that |b1|=60, |b2|=20
 s1, s2 = train_test_split(sprites, test_size=0.2) #split sprites such that |s1|=75, |s2|=25
 trainData,trainY=generateData(b1,s1,trainSize,2,4,0,1,0)
 valData,valY=generateData(b2,s2,testSize,2,4,0,1,0)
 print("generated data")
 print(trainData.shape)
 print(valData.shape)
 print(trainY.shape)
 print(valY.shape)

 out,model=createModel(trainData,trainY,valData,valY)
 print(out)
 print(model)

main()