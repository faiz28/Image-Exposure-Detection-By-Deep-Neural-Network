from pickletools import optimize
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from prepare_img_data import prepareImg
import numpy as np
import random

DIR = './validation/'


def main():
    model = build_model()
    trainX, trainY, testX, testY = preprocess_data()

    print(trainX, trainY, testX, testY)
    # model.compile(loss='mse',optimizer = 'rmsprop')
    # model.fit()
    # model.predict()
    # model.evaluate()

    model.compile(loss='mse', optimizer='rmsprop', metrics='accuracy')
    model.fit(trainX, trainY, epochs=2)
    model.evaluate(testX, testY)


def preprocess_data():
    normal_img = DIR + 'Normal/'
    normal_img = prepareImg(normal_img)
    print(normal_img.shape)
    m = normal_img.shape[0]
    OverExposed_img = DIR + 'OverExposed/'
    OverExposed_img = prepareImg(OverExposed_img)
    print(OverExposed_img.shape)
    n = OverExposed_img.shape[0]

    UnderExposed_img = DIR + 'UnderExposed/'
    UnderExposed_img = prepareImg(UnderExposed_img)
    print(UnderExposed_img.shape)
    n = UnderExposed_img.shape[0]

    imgSet = np.concatenate(
        (normal_img, OverExposed_img, UnderExposed_img), axis=0)
    print(imgSet.shape)
    # preprocess image data
    print(imgSet.min(), imgSet.max())
    imgSet = preprocess_input(imgSet)
    print(imgSet.min(), imgSet.max())

    # prepare lables
    labelSet1 = np.zeros(m, dtype=np.uint8)
    labelSet2 = np.ones(n, dtype=np.uint8)
    lableSet = np.concatenate((labelSet1, labelSet2))
    print(lableSet)

    # trun label data into one hot vector
    lableSet = to_categorical(lableSet)

    # suffuel image data
    n = imgSet.shape[0]
    indices = np.arange(n)
    print(indices)
    random.shuffle(indices)
    print(indices)
    imgSet = imgSet[indices]
    labelSet = lableSet[indices]
    trainX = imgSet[:int(n*0.5)]
    trainY = lableSet[:int(n*0.5)]

    testX = imgSet[int(n*0.5):]
    testY = lableSet[int(n*0.5):]
    print("done")
    return trainX, trainY, testX, testY


def build_model():
    baseModel = VGG16(input_shape=(256, 256, 3), include_top=False)
    baseModel.summary()

    for layer in baseModel.layers:
        layer.trainable = False
    baseModel.summary()

    inputs = baseModel.input
    x = baseModel.output
    x = Flatten()(x)
    x = Dense(16, activation='sigmoid')(x)
    outputs = Dense(2, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.summary()
    return model


if __name__ == '__main__':
    main()
