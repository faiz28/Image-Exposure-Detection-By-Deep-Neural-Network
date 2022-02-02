import os
import cv2
import numpy as np
DIR = './'
Img_size = (256, 256)


def main():
    imgDir = DIR + './Elephant/'
    img_ele = prepareImg(imgDir)

    # print(imgList)
    imgDir = DIR + './Baby/'
    img_ele = prepareImg(imgDir)


def prepareImg(imgDir):
    imgSet = []
    imgList = os.listdir(imgDir)
    n = len(imgList)
    for i in range(n):
        imgPath = imgDir + imgList[i]
        if os.path.exists(imgPath):
            # print(imgPath)
            img = cv2.imread(imgPath)
            # print(img.shape)
            resizedImg = cv2.resize(img, Img_size)
            # print(resizedImg.shape)
            rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
            imgSet.append(rgbImg)
        else:
            print("path doesn't exists")

    # print(len(imgSet))
    imgSet = np.array(imgSet, dtype=np.uint8)
    # print(imgSet.shape)
    return imgSet


if __name__ == "__main__":
    main()
