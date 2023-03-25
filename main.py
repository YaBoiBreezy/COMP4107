#Alexander Breeze   101 143 291
#Michael Balcerzak  101 071 699
#Ifeanyichukwu Obi  101 126 269


'''
sources:
https://stackoverflow.com/questions/62756658/loss-function-for-yolo
https://machinelearningspace.com/yolov3-tensorflow-2-part-1/
https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
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
import scipy
from scipy import cluster
from scipy.cluster.hierarchy import dendrogram, linkage

gridSize=16 #divide image into gridSize x gridSize quadrants
quadSize=256/gridSize #number of pixels in each quadrant

def imageSimilarity(i1,i2):
 #histogram looks at popularity of different colors, good but not perfect necessarily
 #recommend compressing to dense vector representation  https://github.com/UKPLab/sentence-transformers
 #opencv feature matching   https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
 return random.random()*10

def grouping(image,boxes,confidences):
 print(boxes.shape)
 print(confidences.shape)
 images=[]
 threshold=0.5
 dim=32
 #get image from box, for all boxes with confidence>threshold, reshape to dim x dim, add to images list
 for bigX in range(gridSize):
  for bigY in range(gridSize):
   box=boxes[bigX][bigY]
   confidence=confidences[bigX][bigY]
   if confidence>threshold:
    x=int(box[0]+bigX*quadSize)
    y=int(box[1]+bigY*quadSize)
    w=int(box[2]/2) #halfWidth, halfHeight
    h=int(box[3]/2)
    tmp=image.crop([x-w,y-h,x+w,y+h]).resize((dim,dim))
    tmp=tmp.getdata()
    images.append(tmp)
 print(images)

 if len(images)<2:
  return np.array([[0]*gridSize]*gridSize)

 #construct distance matrix between each pair of images
 sim=[]
 for i in range(len(images)):
  for j in range(i+1,len(images)):
   sim.append(imageSimilarity(images[i],images[j]))
 print(f'similarities: {sim}')

 #group images into groups where the distance between groups is >=threshold
 linkage_matrix = linkage(sim, "single")
 cut=cluster.hierarchy.cut_tree(linkage_matrix, height=[0.5])
 print(f'groups: {cut}')

 #construct group matrix
 groups=np.array([[0]*gridSize]*gridSize)
 index=0
 for bigX in range(gridSize):
  for bigY in range(gridSize):
   box=boxes[bigX][bigY]
   confidence=confidences[bigX][bigY]
   if confidence>threshold:
    groups[bigX][bigY]=cut[index]+1
    index+=1

 return groups

def customLoss(y,yhat):
 #print("SHAPES")
 #print(y.shape)
# boxY=tf.slice(y,[0,0,0,1],[1,gridSize,gridSize,4]) #This should make it so we don't need the useless neurons in the output layer
 #print(boxY.shape)
 a = y[:,:,:,:]*y[:,:,:,:1] #boxY not y
 #print(a.shape)
 #print(yhat.shape)
 b = yhat[:,:,:,:]*y[:,:,:,:1]
 #print(b.shape)
 loss = keras.losses.MeanSquaredError()(a,b) #compare coordinates, but if y_actual[:,:,0]=0 then it should be 0 bc there is no object centered in that quadrant
 return loss

def customLoss2(y,yhat):
 #print("SHAPES2")
 #print(y.shape)
 #print(yhat.shape)
 loss = keras.losses.BinaryCrossentropy()(y,yhat) #compare prediction to actual for whether there is an item in this quad
 return loss


# A function that implements a keras model with the sequential API
def createModel(trainX, trainBox, trainConf, valX, valBox, valConf):
 print("creating model")
 input = keras.Input(shape=(256, 256, 3))
 layer_2 = layers.Conv2D(32, kernel_size=3, padding="same", activation='sigmoid')(input)
 layer_3 = layers.Conv2D(32, kernel_size=32, padding="same", strides=16, activation='sigmoid')(layer_2)
 layer_4 = layers.Conv2D(32, kernel_size=32, padding="same", activation='sigmoid')(layer_3)

 
 final = layers.Conv2D(32, kernel_size=32, padding="same", activation='sigmoid')(layer_4)
 output_1=layers.Conv2D(4, kernel_size=4, padding="same", activation='linear')(final) #bounding box, first value is ignored so loss works
 output_2=layers.Conv2D(1, kernel_size=4, padding="same", activation='sigmoid')(final) #confidence

 model = keras.Model(inputs=input, outputs=[output_1, output_2])
 model.compile(optimizer='adam', loss=[customLoss,customLoss2])
 print(model.summary())
 out = model.fit(x=trainX, y=[trainBox, trainConf], validation_data=[valX, [valBox, valConf]], epochs=10)

 return model


#gets color corresponding to group, but can also return black if too many groups
def getColor(i):
 colordict=["none","red","blue","purple","orange","yellow"]
 if i>=0 and i<6:
  return colordict[i]
 return "black"

#draws colored boxes on image
def drawLabelGroup(image,boxes,groups):
 print(boxes.shape)
 print(groups.shape)
 print(image)

 image2=ImageDraw.Draw(image)
 for bigX in range(gridSize):
  for bigY in range(gridSize):
   box=boxes[bigX][bigY]
   group=groups[bigX][bigY]
   if group:                       #if no confidence, then group[][]=0=False
    x=int(box[0]+bigX*quadSize)
    y=int(box[1]+bigY*quadSize)
    w=int(box[2]/2) #halfWidth, halfHeight
    h=int(box[3]/2)
    image2.rectangle([x-w,y-h,x+w,y+h], fill=None, outline=getColor(group), width=5)
 image.show()


#takes 2 arrays of img, returns array of generated img and array of corresponding sprite img
def generateData(backgroundList,spriteList,numInstances,minSpriteCount,maxSpriteCount,minSpriteTypeCount,maxSpriteTypeCount,rotation,sizing,flipping):
 data=[] #array of compound images
 boxes=[] #array of location of sprite in given quadrant
 confidences=[] #array of confidence of sprite in given quadrant
 groups=[] #categorical array of type of sprite in given quadrant of image
 for _ in range(numInstances):
  newBox=[[[0,0,0,0] for __ in range(gridSize)] for _ in range(gridSize)]
  newConf=[[0 for __ in range(gridSize)] for _ in range(gridSize)]
  newGroups=[[0]*gridSize]*gridSize
  sprites=random.sample(spriteList,random.randint(minSpriteTypeCount,maxSpriteTypeCount))
  compound=random.choice(backgroundList).copy()
  for _ in range(random.randint(minSpriteCount,maxSpriteCount)):
   spriteIndex=random.randint(0,len(sprites)-1)
   tempSprite=sprites[spriteIndex].copy()
   if rotation:
    tempSprite=tempSprite.rotate(random.randint(0,360), expand=1) #true makes it resize to fit new image. Uses nearest neighbor to keep pixel colors
   if sizing:
    newSize=random.randint(16,128)
    tempSprite.thumbnail((newSize,newSize),PIL.Image.NEAREST)
   if flipping and random.randint(0,1)==0:
    tempSprite=tempSprite.transpose(Image.FLIP_LEFT_RIGHT)
   spriteWidth,spriteHeight=tempSprite.size
   spriteX=random.randint(0,256-spriteWidth)
   spriteY=random.randint(0,256-spriteHeight)
   midX=int(spriteX+(spriteWidth/2))
   midY=int(spriteY+(spriteHeight/2))
   bigX=int(midX / quadSize)
   bigY=int(midY / quadSize)
   xOffset=int(midX % quadSize)
   yOffset=int(midY % quadSize)
   #print(f'{quadSize} {spriteWidth} {spriteHeight} {spriteX} {spriteY} {midX} {midY} {bigX} {bigY} {xOffset} {yOffset}')
   tempSprite.convert("RGBA")
   compound.paste(tempSprite, (spriteX,spriteY), tempSprite) #last argument is to apply transparent background
   newBox[bigX][bigY]=[xOffset, yOffset, spriteWidth, spriteHeight]
   newConf[bigX][bigY]=1
   newGroups[bigX][bigY]=spriteIndex+1
  data.append(compound.getdata())
  boxes.append(newBox)
  confidences.append(newConf)
  groups.append(newGroups)
  #compound=drawLabels(compound,np.array(newBoxes).reshape((gridSize**2*4)),np.array(newGroups))
  #compound.show()
 data=np.array(data).reshape((numInstances,256,256,3))
 boxes=np.array(boxes).astype(float)
 confidences=np.array(confidences).astype(float)
 groups=np.array(groups)
 return data, boxes, confidences, groups

#takes the locations of 2 folders of images, returns 2 numpy arrays of those images
def readData():
 backgrounds = []
 for x in range(1,80):
  backgrounds.append(Image.open("./backgrounds/background"+str(x)+".jpg").resize((256,256)))
 sprites = []
 for x in range(100):
  img=Image.open("./sprites/sprite"+str(x)+".png")
  img.thumbnail((128,128),PIL.Image.LANCZOS)
  sprites.append(img)
 return backgrounds, sprites

def main():
 trainSize=8
 valSize=2
 print("reading data")
 backgrounds,sprites=readData()
 print("splitting data")
 b1, b2 = train_test_split(backgrounds, test_size=0.25) #split backgrounds such that |b1|=60, |b2|=20
 s1, s2 = train_test_split(sprites, test_size=0.2) #split sprites such that |s1|=80, |s2|=20
 print("generating datapoints")
 trainData,trainBox,trainConf,trainGroups=generateData(b1,s1,trainSize,2,4,2,3,0,1,0)
 valData,valBox,valConf,valGroups=generateData(b2,s2,valSize,2,4,1,2,0,1,0)
 print("training model")

 username="Michael"
 if username=="Michael":
  model=createModel(trainData,trainBox,trainConf,valData,valBox,valConf)
  boxes,confidences=model.predict(valData)
  groups=grouping(Image.fromarray(np.uint8(valData[0])),boxes[0],confidences[0])
  drawLabelGroup(Image.fromarray(np.uint8(valData[0])),valBox[0],groups)
 elif username=="Patrick":
  groups=grouping(Image.fromarray(np.uint8(valData[0])),valBox[0],valConf[0])
  drawLabelGroup(Image.fromarray(np.uint8(valData[0])),valBox[0],groups)

main()