#Alexander Breeze   101 143 291
#Michael Balcerzak  101 071 699
#Ifeanyichukwu Obi  101 126 269

#TODO:
#make cnn to do yolo
#make loss take 2 outputs, do loss
#make function to take sprites using CNN output, generate data for kmeans
#make function to run kmeans, return bounding boxes and groups

#TOFIX: readdata and generatedata both have small values for faster testing


'''
sources:
https://stackoverflow.com/questions/62756658/loss-function-for-yolo
https://machinelearningspace.com/yolov3-tensorflow-2-part-1/
'''

import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras, linalg, map_fn
from keras import layers
from keras.models import Sequential, Functional
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

def kmeans(image,boxes,confidences):
 images=[]
 threshold=0.5
 dim=64
 for bigX in range(gridSize):
  for bigY in range(gridSize):
   confidence=confidences[bigX*quadSize+bigY]
   box=boxes[int(bigX*quadSize+bigY)*4]
   #get image from box, for all boxes with confidence>threshold, reshape to dim x dim, add to images list
 #call kmeans on images list, create group vector in same format is confidences (list of 0 and int corresponding to boxes)
 return groups

def customLoss(y,yhat):
 print("SHAPES")
 print(y.shape)
 boxY=tf.slice(y,[0,0,0,1],[1,gridSize,gridSize,4]) #This should make it so we don't need the useless neurons in the output layer
 print(boxY.shape)
 a = boxY[:,:,:,:]*y[:,:,:,:1]
 print(a.shape)
 print(yhat.shape)
 b = yhat[:,:,:,:]*y[:,:,:,:1]
 print(b.shape)
 loss = keras.losses.MeanSquaredError()(a,b) #compare coordinates, but if y_actual[:,:,0]=0 then it should be 0 bc there is no object centered in that quadrant
 return loss

def customLoss2(y,yhat):
 print("SHAPES2")
 print(y.shape)
 print(yhat.shape)
 loss = keras.losses.BinaryCrossentropy()(y,yhat) #compare prediction to actual for whether there is an item in this quad
 return loss


# A function that implements a keras model with the sequential API
def createModel(xTrain, yTrain, xVal, yVal):
 print("creating model")
 input = keras.Input(shape=(1024, 1024, 3))
 layer_2 = layers.Conv2D(32, kernel_size=3, padding="same", activation='sigmoid')(input)
 layer_3 = layers.Conv2D(32, kernel_size=32, padding="same", strides=16, activation='sigmoid')(layer_2)
 layer_4 = layers.Conv2D(32, kernel_size=32, padding="same", activation='sigmoid')(layer_3)

 #edit layers here, don't touch layers input, layer_2, layer_3, final, output_1, output_2. NO STRIDES, padding="same" for all
 #make sure final is connected to the previous layer
 
 final = layers.Conv2D(32, kernel_size=32, padding="same", activation='sigmoid')(layer_4)
 output_1=layers.Conv2D(1, kernel_size=4, padding="same", strides=4, activation='sigmoid')(final) #confidence
 output_2=layers.Conv2D(4, kernel_size=4, padding="same", strides=4, activation='linear')(final) #bounding box, first value is ignored so loss works

 model = keras.Model(inputs=input, outputs=[output_1, output_2])
 model.compile(optimizer='adam', loss=[customLoss2,customLoss])
 print(model.summary())
 yTconf=yTrain[:,:,:,0]
 yVconf=yVal[:,:,:,0]
 print(yTrain.shape)
 print(yTconf.shape)
 print(yVal.shape)
 print(yVconf.shape)
 out = model.fit(x=xTrain, y=[yTconf, yTrain], validation_data=[xVal, [yVconf, yVal]], epochs=10)

 return model


def drawLabels(image,boxes,groups):
 print(boxes.shape)
 print(groups.shape)
 colordict=["red","blue","purple","orange","yellow"]
 image2=ImageDraw.Draw(image)
 for bigX in range(gridSize):
  for bigY in range(gridSize):
   group=groups[bigX][bigY]
   box=boxes[bigX][bigY]
   if group:
    x=int(box[0]+bigX*quadSize)
    y=int(box[1]+bigY*quadSize)
    w=int(box[2]/2) #halfWidth, halfHeight
    h=int(box[3]/2)
    image2.rectangle([x-w,y-h,x+w,y+h], fill=None, outline=colordict[group], width=3)
 return image



#takes 2 arrays of img, returns array of generated img and array of corresponding sprite img
def generateData(backgroundList,spriteList,numInstances,minSpriteCount,maxSpriteCount,minSpriteTypeCount,maxSpriteTypeCount,rotation,sizing,flipping):
 data=[] #array of compound images
 y=[] #array of confidence and location of sprite in given quadrant
 groups=[] #categorical array of type of sprite in given quadrant of image
 for _ in range(numInstances):
  newY=[[[0,0,0,0,0] for __ in range(gridSize)] for _ in range(gridSize)]
  newGroups=[[0]*gridSize]*gridSize
  sprites=random.sample(spriteList,random.randint(minSpriteTypeCount,maxSpriteTypeCount))
  compound=random.choice(backgroundList).copy()
  for _ in range(random.randint(minSpriteCount,maxSpriteCount)):
   spriteIndex=random.randint(0,len(sprites)-1)
   tempSprite=sprites[spriteIndex].copy()
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
   midY=int(spriteY+(spriteHeight/2))
   bigX=int(midX / quadSize)
   bigY=int(midY / quadSize)
   xOffset=int(midX % quadSize)
   yOffset=int(midY % quadSize)
   #print(f'{quadSize} {spriteWidth} {spriteHeight} {spriteX} {spriteY} {midX} {midY} {bigX} {bigY} {xOffset} {yOffset}')
   tempSprite.convert("RGBA")
   compound.paste(tempSprite, (spriteX,spriteY), tempSprite) #last argument is to apply transparent background
   newY[bigX][bigY]=[1, xOffset, yOffset, spriteWidth, spriteHeight]
   newGroups[bigX][bigY]=spriteIndex+1
  data.append(compound.getdata())
  y.append(newY)
  groups.append(newGroups)
  #compound=drawLabels(compound,np.array(newBoxes).reshape((gridSize**2*4)),np.array(newGroups))
  #compound.show()
 data=np.array(data).reshape((numInstances,1024,1024,3))
 y=np.array(y).astype(float)
 groups=np.array(groups)
 return data, y, groups

#takes the locations of 2 folders of images, returns 2 numpy arrays of those images
def readData():
 backgrounds = []
 for x in range(1,8):
  backgrounds.append(Image.open("./backgrounds/background"+str(x)+".jpg").resize((1024,1024)))
 sprites = []
 for x in range(10):
  img=Image.open("./sprites/sprite"+str(x)+".png")
  img.thumbnail((512,512),PIL.Image.LANCZOS)
  sprites.append(img)
 return backgrounds, sprites

def main():
 trainSize=8
 testSize=2 
 print("reading data")
 backgrounds,sprites=readData()
 print("splitting data")
 b1, b2 = train_test_split(backgrounds, test_size=0.25) #split backgrounds such that |b1|=60, |b2|=20
 s1, s2 = train_test_split(sprites, test_size=0.2) #split sprites such that |s1|=80, |s2|=20
 print("generating datapoints")
 trainData,trainY,trainGroups=generateData(b1,s1,trainSize,2,4,2,3,0,1,0)
 valData,valY,valGroups=generateData(b2,s2,testSize,2,4,1,2,0,1,0)
 print("training model")

 model=createModel(trainData,trainY,valData,valY)
 boxes,confidences=model.predict(valData[0])
 groups=kmeans(valData[0],boxes,confidences)
 drawLabels(valData[0],boxes,groups)

main()