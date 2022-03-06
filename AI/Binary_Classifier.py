from tensorflow.keras.layers import Dense,Flatten
from Prepare_ImageData import prepare_image_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import random

IMG_W = 32
IMG_H = 32

DIR = '/home/cse/AI_Lab_2021/'

def main():
	model = build_model()
	trainX, trainY, testX, testY = preprocess_data()
	
	model.compile(loss = 'mse', optimizer = 'rmsprop')
	model.fit()
	model.predict()
	model.evaluate()
		
def preprocess_data():
	# Load image data
	imgDir = DIR + 'Elephant/'
	imgSet1 = prepare_image_array(imgDir, IMG_W, IMG_H)
	print(imgSet1.shape)
	m = imgSet1.shape[0]
	
	imgDir = DIR + 'Baby/'
	imgSet2 = prepare_image_array(imgDir, IMG_W, IMG_H)
	print(imgSet2.shape)
	n = imgSet2.shape[0]
	
	# Put all image data into one array.
	imgSet = np.concatenate((imgSet1, imgSet2), axis = 0)
	print(imgSet.shape)
	
	# Preprocess image data to be fit with VGG16
	print(imgSet.max(), imgSet.min())
	imgSet = preprocess_input(imgSet)
	print(imgSet.max(), imgSet.min())
	
	# Prepare labels.
	labelSet1 = np.zeros(m, dtype = np.uint8)
	labelSet2 = np.ones(n, dtype = np.uint8)
	labelSet = np.concatenate((labelSet1, labelSet2))
	print(labelSet)
	
	# Turn label data into on-hot vectors
	labelSet = to_categorical(labelSet)
	print(labelSet)
	
	# Shuffle image data and labels
	n = imgSet.shape[0]
	indices = np.arange(n)
	print(indices)
	random.shuffle(indices)
	print(indices)	
	imgSet = imgSet[indices]
	labelSet = labelSet[indices]
	
	# Split data into training and test sets
	m = int(n*0.5)
	trainX = imgSet[:m]
	trainY = labelSet[:m]
	
	testX = imgSet[m:]
	testY = labelSet[m:]
	
	return trainX, trainY, testX, testY

def build_model():
	baseModel = VGG16(input_shape = (IMG_H, IMG_W, 3), include_top = False) #VGG19() #InceptionResNetV2()
	baseModel.summary()
	
	for layer in baseModel.layers:
		layer.trainable = False
	baseModel.summary()
	
	inputs = baseModel.input
	x = baseModel.output
	x = Flatten()(x)
	x = Dense(16, activation = 'sigmoid')(x)
	outputs = Dense(2, activation = 'sigmoid')(x)
	
	model = Model(inputs, outputs)
	model.summary()
	
	return model

if __name__ == '__main__':
	main()

