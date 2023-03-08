#Alexander Breeze   101 143 291
#Michael Balcerzak  101 071 699
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
import cv2
import glob
import random

def custom_loss(y,yhat): #using lanbda function to have access to yhat while iterating over y
 c1=y #loss is minimum of sum of squares of manhattan distances between pair of points yhat and all pairs of points in y
 c2=yhat
 distances=map_fn(lambda c1: tf.add(tf.square(tf.add(tf.abs(tf.subtract(c1[0],c2[0])),tf.abs(tf.subtract(c1[1],c2[1])))),tf.square(tf.add(tf.abs(tf.subtract(c1[2],c2[2])),tf.abs(tf.subtract(c1[3],c2[3]))))))
 return tf.minimum(distances)

# A function that implements a keras model with the sequential API
def createModelHP(xTrain, yTrain, xVal, yVal, params):
 print("creating model:")
 print(params)
 model = Sequential()
 model.add(layers.Conv2D(filters=params['filter1'], kernel_size=(3,3,3), activation=params['convActivation'], input_shape=(1024, 1024, 3)))
 model.add(layers.MaxPooling2D(params['pool1']))
 model.add(layers.Conv2D(filters=params['filter2'], kernel_size=params['kernel2'], activation=params['convActivation']))
 model.add(layers.MaxPooling2D(params['pool2']))
 model.add(layers.Conv2D(filters=params['filter3'], kernel_size=params['kernel3'], activation=params['convActivation']))
 model.add(layers.MaxPooling2D(params['pool3']))
 model.add(layers.Conv2D(filters=params['filter4'], kernel_size=params['kernel4'], activation=params['convActivation']))
 model.add(layers.MaxPooling2D(params['pool4']))
 model.add(layers.Conv2D(filters=params['filter5'], kernel_size=params['kernel5'], activation=params['convActivation']))
 model.add(layers.MaxPooling2D(params['pool5']))
 model.add(layers.Flatten())
 model.add(layers.Dense(params['neuron1'], activation=params['NNActivation']))
 model.add(layers.Dense(params['neuron2'], activation=params['NNActivation']))
 model.add(layers.Dense(params['neuron3'], activation=params['NNActivation']))
 model.add(layers.Dense(params['neuron4'], activation=params['NNActivation']))
 model.add(layers.Dense(params['neuron5'], activation=params['NNActivation']))
 model.add(layers.Dense(4, activation='softplus'))

 model.compile(loss=custom_loss, optimizer='adam') #, metrics=['accuracy']
 out = model.fit(x=xTrain, y=yTrain, validation_data=[xVal, yVal], epochs=100, batch_size=params['batch_size'], verbose=0)
 print(out)
 print(model)
 return out,model

# A function that implements a keras model with the sequential API
def createModel(xTrain, yTrain, xVal, yVal):
 print("creating model")
 model = Sequential()
 model.add(layers.Conv2D(512, kernel_size=(3,3,3), activation='sigmoid', input_shape=(1024, 1024, 3)))
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
 model.add(layers.Dense(2048, activation=params['NNActivation']))
 model.add(layers.Dense(2048, activation=params['NNActivation']))
 model.add(layers.Dense(1024, activation=params['NNActivation']))
 model.add(layers.Dense(512, activation=params['NNActivation']))
 model.add(layers.Dense(512, activation=params['NNActivation']))
 model.add(layers.Dense(4, activation='softplus'))

 model.compile(loss=custom_loss, optimizer='adam')
 out = model.fit(x=xTrain, y=yTrain, validation_data=[xVal, yVal], epochs=100, batch_size=params['batch_size'], verbose=0)
 print(out)
 print(model)
 return out,model

def trainModel(x,y):
 return model,loss

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
  compound=random.choice(backgrounds).copy()
  for _ in range(random.randint(minSpriteCount,maxSpriteCount)):
   tempSprite=sprite.copy()
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
 for x in range(len(y)):
  for z in range(maxSpriteCount-len(y[x])):
   y[x].append([-1024,-1024,-1024,-1024]) #filling out y, NN can only output positive coordinates due to a()=softplus
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
 trainSize=80
 testSize=20
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

 #hyperparameters of neural network to optimize
 p = {
    'filter1': [8, 16, 32, 64],
    'filter2': [8, 16, 32, 64],
    'filter3': [8, 16, 32, 64],
    'filter4': [8, 16, 32, 64],
    'filter5': [8, 16, 32, 64],
    'pool1': [2, 3, 4, 5],
    'pool2': [2, 3, 4, 5],
    'pool3': [2, 3, 4, 5],
    'pool4': [2, 3, 4, 5],
    'pool5': [2, 3, 4, 5],
    'neuron1': [512, 1024, 2048, 4096, 8192, 16384],
    'neuron2': [512, 1024, 2048, 4096, 8192, 16384],
    'neuron3': [512, 1024, 2048, 4096, 8192, 16384],
    'neuron4': [512, 1024, 2048, 4096, 8192, 16384],
    'neuron5': [512, 1024, 2048, 4096, 8192, 16384],
    'convActivation': ['sigmoid', 'elu', 'linear', 'softmax'],
    'NNActivation': ['sigmoid', 'elu', 'linear', 'softmax'],
    'epochs' : [50, 100, 200],
    'batch_size': (10,trainSize,10)}  #(min,max,steps)
 out,model=createModel(trainData,trainY,valData,valY)
 print(out)
 print(model)

main()