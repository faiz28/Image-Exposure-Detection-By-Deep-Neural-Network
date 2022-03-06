from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array, array_to_img
import cv2
import matplotlib.pyplot as plt

import numpy as np
m = 32
n = 32
c = 2


def build_model():
    baseModel = VGG16(input_shape=(m, n, 3), include_top=False)
    baseModel.summary()

    count = 0
    for layer in baseModel.layers:
        layer.trainable = False

    baseModel.summary()

    inputs = baseModel.input
    x = baseModel.output
    x = Flatten()(x)

    x = Dense(units=4096, activation="relu")(x)
    x = Dense(units=128, activation="relu")(x)
    outputs = Dense(units=c, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.summary()

    return model


def prepare_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()

    # select 1,2 from mnist dataset
    train_indices = np.argwhere(np.logical_and(trainY >= 1, trainY <= 2))
    test_indices = np.argwhere(np.logical_and(testY >= 1, testY <= 2))

    trainX = trainX[train_indices.flatten()]
    trainY = trainY[train_indices.flatten()]
    testX = testX[test_indices.flatten()]
    testY = testY[test_indices.flatten()]

    trainX = np.pad(trainX, ((0, 0), (2, 2), (2, 2)), 'constant')
    testX = np.pad(testX, ((0, 0), (2, 2), (2, 2)), 'constant')

    trainX = np.dstack([trainX] * 3)
    testX = np.dstack([testX]*3)

    trainX = trainX.reshape(-1, 32, 32, 3)
    testX = testX.reshape(-1, 32, 32, 3)

    trainY = to_categorical(trainY == 2)
    testY = to_categorical(testY == 2)

    trainX = trainX.astype(np.float32)
    testX = testX.astype(np.float32)
    trainX /= 255
    testX /= 255

    return trainX, trainY, testX, testY


def main():
    trainX, trainY, testX, testY = prepare_data()
    model = build_model()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    model.fit(trainX, trainY, epochs=100, batch_size=50, validation_split=.3)
    score, acc = model.evaluate(testX, testY)
    print("loss : %s, accuracy : %s\n" % (score, acc))


if __name__ == '__main__':
    main()
