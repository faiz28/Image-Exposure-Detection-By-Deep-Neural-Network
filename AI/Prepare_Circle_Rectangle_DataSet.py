# Prepare a synthetic dataset by drawing RGB colored rectangles and circles on a square.
# Sangeeta Biswas
# 17.1.2022

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def main():
	shapeH = 256
	shapeW = 256
	rectangleSet = prepare_rectangleSet(100, shapeH, shapeW)
	circleSet = prepare_circleSet(100, shapeH, shapeW)
		
def prepare_circleSet(n, shapeH, shapeW):
	circleSet = np.zeros((n, shapeH, shapeW, 3), dtype = np.uint8)
	for i in range(n):
		# Point co-ordinates
		x = random.randint(0, 125)
		y = random.randint(0, 150)		
		radius = random.randint(5, 100)
		print(x, y, radius)
		
		# Color
		r = random.randint(0, 255)
		g = random.randint(0, 255)
		b = random.randint(0, 255)
		
		# Draw circle
		cv2.circle(circleSet[i], (x, y), radius, (r, g, b), -1)
		
	# Cross-check
	display_shape(circleSet[:9])
	
	return circleSet 

def prepare_rectangleSet(n, shapeH, shapeW):
	rectangleSet = np.zeros((n, shapeH, shapeW, 3), dtype = np.uint8)
	for i in range(n):
		# Point co-ordinates
		x = random.randint(0, 150)
		y = random.randint(0, 150)		
		w = random.randint(5, 100)
		h = random.randint(5, 100)
		print(x, y, w, h)
		
		# Color
		r = random.randint(0, 255)
		g = random.randint(0, 255)
		b = random.randint(0, 255)
		
		# Draw rectangle
		cv2.rectangle(rectangleSet[i], (x, y), (x + w, y + h), (r, g, b), -1)
		
	# Cross-check
	display_shape(rectangleSet[:9])
	
	return rectangleSet 
	
def display_shape(imgSet):
	plt.figure(figsize = (20, 20))
	for i in range(9):
		plt.subplot(3, 3, i + 1)		
		plt.imshow(imgSet[i])
		plt.axis('off')
	plt.show()
	plt.close()

if __name__ == '__main__':
	main()
