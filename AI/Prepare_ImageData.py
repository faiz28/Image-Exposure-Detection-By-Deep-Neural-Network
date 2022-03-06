# Prepare image data arrays of two classes.
# 19.1.2022

import os
import cv2
import numpy as np

DIR = '/home/cse/AI_Lab_2021/'
IMG_H = 256
IMG_W = 256

def main():
	imgDir = DIR + 'Elephant/'
	prepare_image_array(imgDir)
	
	imgDir = DIR + 'Baby/'
	prepare_image_array(imgDir)
	
def prepare_image_array(imgDir):
	imgList = os.listdir(imgDir)
	print(imgList)
	n = len(imgList)
	
	imgSet = []
	for i in range(n):
		imgPath = imgDir + imgList[i]
		if (os.path.exists(imgPath)):
			print(imgPath)
			
			# Load image.
			img = cv2.imread(imgPath)
			print(img.shape)
			
			# Resize image.
			resizedImg = cv2.resize(img, (IMG_H, IMG_W))
			print(resizedImg.shape)
			
			# Convert BGR image into RGB image.
			rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
			
			# Put image into a list
			imgSet.append(rgbImg)
		else:
			print("It is not a valid image path.")
		
	print(len(imgSet))
	imgSet = np.array(imgSet, dtype = np.uint8)
	print(imgSet.shape)

if __name__ == '__main__':
	main()
