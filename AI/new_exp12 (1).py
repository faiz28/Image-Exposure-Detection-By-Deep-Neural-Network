from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

(trainX, trainY), (testX, testY) = mnist.load_data()

train_indices = np.argwhere((trainY == 2) | (trainY == 3))
train_indices = np.squeeze(train_indices)

test_indices = np.argwhere((testY == 2) | (testY == 3))
test_indices = np.squeeze(test_indices)

trainX = trainX[train_indices]
trainY = trainY[train_indices]

testX = testX[test_indices]
testY = testY[test_indices]

trainX = trainX.astype(np.float32)
trainX /= 255

testX = testX.astype(np.float32)
testX /= 255

trainY = to_categorical(trainY == 3)
testY = to_categorical(testY == 3)

trainX = np.pad(trainX, ((0,0),(2,2),(2,2)), 'constant')
testX = np.pad(testX, ((0,0),(2,2),(2,2)), 'constant')

trainX = np.stack((trainX,)*3, axis=-1)
testX = np.stack((testX,)*3, axis=-1)

m = trainX.shape[1]
n = trainX.shape[2]
h = 4
c = 2
base_model = VGG16(input_shape=(m, n, 3), include_top=False)
for layer in base_model.layers:
    layer.trainable = False

input_layer = base_model.input
x = Flatten()(input_layer)
output_layer = Dense(c)(x)
model = Model(input_layer, output_layer)

model.compile(loss='mse', optimizer='rmsprop', metrics='accuracy')
model.fit(trainX, trainY, epochs=5, validation_split=0.2)
model.evaluate(testX, testY)
