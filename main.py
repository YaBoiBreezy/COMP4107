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

def customLoss(y,yhat): #using lambda function to have access to yhat while iterating over y
 a = keras.losses.BinaryCrossentropy()(y_actual[:,:,0],y_pred[:,:,0]) #compare prediction to actual for whether there is an item in this quad
 b = keras.losses.MeanSquaredError()(y_actual[:,:,1:]*y_actual[:,:,0], y_pred[:,:,1:]*y_actual[:,:,0]) #compare coordinates, but if y_actual[:,:,0]=0 then it should be 0 bc there is no object centered in that quadrant
 loss= a + b
 return loss

# A function that implements a keras model with the sequential API
def createModel(xTrain, yTrain, xVal, yVal):
 print("creating model")
 input = keras.Input(shape=(1024, 1024, 3))
 layer_2 = layers.Conv2D(32, kernel_size=32, strides=16, activation='sigmoid')(input)
 layer_3 = layers.Conv2D(32, kernel_size=32, strides=16, activation='sigmoid')(layer_2)
 layer_4 = layers.Flatten()(layer_3)
 layer_5 = layers.Dense(256, activation='sigmoid')(layer_4)
 
 outputs=[]
 for x in range(gridSize):
  outputs.append([])
  for y in range(gridSize):
   output_1=layers.Dense(1, activation='sigmoid') #confidence
   output_2=layers.Dense(4, activation='linear')  #bounding box
   concat_out=layers.Concatenate([output_1, output_2])
   outputs[-1].append([concat_out])

 model = keras.Model(inputs=input, outputs=outputs)
 model.compile(optimizer='adam', loss=customLoss)
 print(model.summary())
 out = model.fit(x=xTrain, y=yTrain, validation_data=[xVal, yVal], epochs=100)

 input = Input(shape = (X_train.shape[1]))
 branchA = Dense(neuronsA, activation = "relu")(input)
 branchB = Dense(neuronsB, activation = "sigmoid")(input)
 out = concatenate([branchA, branchB])

 return model

def drawLabels(image,boxes,groups):
 colordict=["red","blue","purple","orange","yellow"]
 image2=ImageDraw.Draw(image)
 for bigX in range(gridSize):
  for bigY in range(gridSize):
   group=groups[bigX*quadSize+bigY]
   box=boxes[bigX*quadSize+bigY]
   if group[0]:
    cords=centerToCorner(box,bigX*quadSize,bigY*quadSize)
    image2.rectangle(cords, fill=None, outline=colorDict[group], width=3)
 return image

def centerToCorner(cords,offsetX,offsetY):
 x=int(box[0]+offsetX)
 y=int(box[1]+offsetY)
 w=int(box[2]/2) #halfWidth, halfHeight
 h=int(box[3]/2)
 return [x-w,y-h,x+w,y+h]

def cornerToCenter(cords):
 w=int(cords[2]-cords[0])
 h=int(cords[3]-cords[1])
 cx=int(cords[0]+w/2)
 cy=int(cords[1]+h/2)
 return [cx,cy,w,h]

spriteWidth,spriteHeight=tempSprite.size
   spriteX=random.randint(0,1024-spriteWidth)
   spriteY=random.randint(0,1024-spriteHeight)
   #midX=int(spriteX+(spriteWidth/2))
   #midY=int(spriteY+(spriteHeight/2))
   bigX=int(midX / quadSize)
   bigY=int(midY / quadSize)
   #xOffset=int(midX % quadSize)
   #yOffset=int(midY % quadSize)
   # print(f'{quadSize} {spriteWidth} {spriteHeight} {spriteX} {spriteY} {midX} {midY} {bigX} {bigY} {xOffset} {yOffset}')
   tempSprite.convert("RGBA")
   compound.paste(tempSprite, (spriteX,spriteY), tempSprite) #last argument is to apply transparent background


#takes 2 arrays of img, returns array of generated img and array of corresponding sprite img
def generateData(backgroundList,spriteList,numInstances,minSpriteCount,maxSpriteCount,minSpriteTypeCount,maxSpriteTypeCount,rotation,sizing,flipping):
 data=[] #array of compound images
 boxes=[] #array of sprite locations in each compound image
 confidences=[] #categorical array of type of sprite in given quadrant of image
 groups=[] #binary array of if there is a sprite in given quadrant of image
 for _ in range(numInstances):
  newBoxes=[[[0,0,0,0] for __ in range(gridSize)] for _ in range(gridSize)]
  newconfidences=[0]*grid_size**2
  newgroups=[0]*grid_size**2
  sprites=random.sample(spriteList,random.randint(minSpriteTypeCount,maxSpriteTypeCount))
  compound=random.choice(backgrounds).copy()
  for _ in range(random.randint(minSpriteCount,maxSpriteCount)):
   spriteIndex=random.randint(0,len(sprites))
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
   #midX=int(spriteX+(spriteWidth/2))
   #midY=int(spriteY+(spriteHeight/2))
   bigX=int(midX / quadSize)
   bigY=int(midY / quadSize)
   #xOffset=int(midX % quadSize)
   #yOffset=int(midY % quadSize)
   # print(f'{quadSize} {spriteWidth} {spriteHeight} {spriteX} {spriteY} {midX} {midY} {bigX} {bigY} {xOffset} {yOffset}')
   tempSprite.convert("RGBA")
   compound.paste(tempSprite, (spriteX,spriteY), tempSprite) #last argument is to apply transparent background
   newLabel[bigX][bigY]=cornerToCenter([spriteX,spriteY,spriteWidth,spriteHeight])  #[xOffset, yOffset, spriteWidth, spriteHeight]
   newConfidences[bigX*quadSize+bigY]=1
   newGroups[bigX*quadSize+bigY]=spriteIndex+1
  data.append(compound.getdata())
  boxes.append(newBoxes)
  confidences.append(newConfidences)
  groups.append(newGroups)
  # compound=drawRectangles(compound,newLabel)
  # compound.show()
 data=np.array(data).reshape((numInstances,1024,1024,3))
 boxes=np.array(boxes).reshape((numInstances,(gridSize**2)*5))
 confidences=np.array(confidences).reshape((numInstances,(gridSize**2)*5))
 groups=np.array(groups).reshape((numInstances,(gridSize**2)*5))
 y={'boxes':boxes;'confidences':confidences;'groups':groups}
 return data, y

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
 trainData,trainY=generateData(b1,s1,trainSize,2,4,2,3,0,1,0)
 valData,valY=generateData(b2,s2,testSize,2,4,2,3,0,1,0)
 print("training model")
 print(trainData.shape)
 print(valData.shape)
 print(trainY.shape)
 print(valY.shape)

 model=createModel(trainData,trainY,valData,valY)
 boxes,labels=model.predict(valData[0])
 boxes,groups=kmeans(valData[0],boxes,labels)
 drawLabels(valData[0],boxes,groups)

main()