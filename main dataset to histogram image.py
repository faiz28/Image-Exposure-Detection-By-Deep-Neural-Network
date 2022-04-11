import os
import cv2
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt
DIR = './Main Database/'
destination = "./histogram/"
count =0

def main():
    name = "Normal/"
    imgDir = DIR + name
    imgList = os.listdir(imgDir)
    for imgFile in imgList:
        imgPath = imgDir + imgFile
        imgSet = load_img2(imgPath,name)
    name = "OverExposed/"
    imgDir = DIR + name
    imgList = os.listdir(imgDir)
    for imgFile in imgList:
        imgPath = imgDir + imgFile
        imgSet = load_img2(imgPath,name)
    name = "UnderExposed/"
    imgDir = DIR + name
    imgList = os.listdir(imgDir)
    for imgFile in imgList:
        imgPath = imgDir + imgFile
        imgSet = load_img2(imgPath,name)
def load_img2(imgPath,name):
    global count
    print(count)
    bgrImg = cv2.imread(imgPath)
    grayImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2GRAY)
    plt.hist(grayImg.ravel(), bins=256, range=(0, 256), color = 'k') #calculating histogram
    plt.savefig(destination + name +imgPath.split('/')[-1].split('.')[0] + '.jpg')
    count+=1
    plt.clf()


if __name__ == '__main__':
	main()
