import os
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import random
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt
TRAIN_SPLIT = 0.85

DIR = './'
imgH = 64
imgW = 64

total_img = 5

# file = open('data_index.txt', 'w')
count = 0

def load_image_data():
    imgDir = DIR + 'Normal/'
    imgSet1 = prepare_image_array(imgDir, imgW, imgH)
    m = imgSet1.shape[0]

    imgDir = DIR + 'OverExposed/'
    imgSet2 = prepare_image_array(imgDir, imgW, imgH)
    n = imgSet2.shape[0]

    imgDir = DIR + 'UnderExposed/'
    imgSet3 = prepare_image_array(imgDir, imgW, imgH)
    o = imgSet3.shape[0]

    # Put all image data into one array.
    imgSet = np.concatenate((imgSet1, imgSet2, imgSet3), axis=0)
    print(imgSet.shape)

    labelSet1 = np.zeros(m, dtype=np.uint8)
    labelSet2 = np.ones(n, dtype=np.uint8)
    label_o = np.ones(o, dtype=np.uint8)
    labelSet3 = np.add(label_o, label_o)
    labelSet = np.concatenate((labelSet1, labelSet2, labelSet3), axis=0)


    p = imgSet.shape[0]  # p = n + m + o
    # indices = np.array(file.read())
    # indices = np.arange(p)
    # random.shuffle(indices)
    # for i in range(len(indices)):
    #     file.write(str(indices[i])+",")
    
    # file.close()
    f = open('data_index.txt', 'r')
    indices = []
    for x in f.read().split(','):
        if len(x) > 0:
            num = ''
            for j in x:
                if(j>='0' and j<='9'):
                    num +=j;
            if len(num)>0 and  int(num)<p:
                indices.append(int(num)) 
   
    print(indices)
    imgSet = imgSet[indices]
    labelSet = labelSet[indices]
    print(labelSet[:5])
    print("Label set : ")
    labelSet = labelSet.reshape(p, 1)

#   one hot encoding

    labelSet = to_categorical(labelSet)

    # print(labelSet[:5])
    
    # print(imgSet[0])
    imgSet = imgSet.astype(np.float32)
    imgSet = imgSet/255.0

    r = int(p * TRAIN_SPLIT)
    trainX = imgSet[:r]
    trainY = labelSet[:r]
    testX = imgSet[r:]
    testY = labelSet[r:]

    return trainX, trainY, testX, testY



def prepare_image_array(imgDir, imgW, imgH):
    imgList = os.listdir(imgDir)
    # print(imgList)
    n = len(imgList)

    imgSet = []
    for i in range(n):
        imgPath = imgDir + imgList[i]
        if (os.path.exists(imgPath)):
            # print(imgPath)
            img = cv2.imread(imgPath)
            resizedImg = cv2.resize(img, (imgW, imgH))
            rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
            imgSet.append(rgbImg)
        else:
            print("It is not a valid image path.")

    # print("total image "+str(len(imgSet)))
    imgSet = np.array(imgSet, dtype=np.uint8)
    # print("total shape "+str(imgSet.shape))

    return imgSet

