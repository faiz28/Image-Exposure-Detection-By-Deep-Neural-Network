from cgi import test

from sklearn import metrics
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D
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

m = trainX.shape[1]
n = trainX.shape[2]
h = 4
c = 2
input_layer = Input((m, n, 1))
x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
for i in range(h - 1):
    x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
output_layer = Dense(c)(x)
model = Model(input_layer, output_layer)

model.compile(loss='mse', optimizer='rmsprop', metrics='accuracy')
model.fit(trainX, trainY, epochs=5, validation_split=0.2)
model.evaluate(testX, testY)
