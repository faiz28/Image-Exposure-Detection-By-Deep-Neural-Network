# Faiz ahmed
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, History
import random
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tkinter
matplotlib.use('TkAgg')

DIR = './train/'
TRAIN_SPLIT = 0.85
imgH = 32
imgW = 32


def main():
    trainX, trainY, testX, testY = preprocess_data()
    modelPath = DIR + 'VGG_BinaryClassifier.hdf5'
    training(trainX, trainY, modelPath)
    testing(testX, testY, modelPath)


def testing(testX, testY, modelPath):
    testXX = preprocess_input(testX)
    model = load_model(modelPath)
    predictions = model.predict(testXX)
    print(predictions)
    predictedClass = []
    for a in predictions:
        a = max(a)
        if a < 0.33:
            predictedClass.append("Normal")
        elif a < 0.66:
            predictedClass.append("OverExposed")
        else:
            predictedClass.append("UnderExposed")
    print('Predicted Class: {}'.format(predictedClass))
    display_images_with_predictions(testX, predictedClass)
    model.compile(metrics='accuracy')
    loss, accuracy = model.evaluate(testXX, testY)
    print('Accuracy: {}'.format(accuracy))


def display_images_with_predictions(testX, predictedClass):
    plt.figure(figsize=(20, 20))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.title(predictedClass[i])
        plt.imshow(testX[i])
        plt.axis('off')
    # plt.show()
    plt.legend()
    plt.savefig(DIR + 'test_data.png')
    plt.close()


def training(trainX, trainY, modelPath):
    trainX = preprocess_input(trainX)
    print(trainY)
    model = build_model()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt)
    callbackList = [EarlyStopping(monitor='val_loss', patience=20), History()]
    history = model.fit(trainX, trainY, epochs=20, batch_size=64,
                        validation_split=0.2, callbacks=callbackList)

    model.save(modelPath)
    plot_loss_acc(history)


def plot_loss_acc(history):
    loss = history.history['loss']
    valLoss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(20, 20))
    plt.rcParams['font.size'] = '14'
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, valLoss, 'k*-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(DIR + 'training_validation_loss.png')
    plt.close()


def build_model():
    baseModel = VGG16(input_shape=(imgH, imgW, 3), include_top=False)
    baseModel.summary()

    for layer in baseModel.layers:
        layer.trainable = False
    baseModel.summary()

    inputs = baseModel.input
    x = baseModel.output
    x = Flatten()(x)
    x = Dense(8, activation='sigmoid')(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.summary()

    return model


def preprocess_data():
    Normal = DIR + 'Normal/'
    OverExposed = DIR + 'OverExposed/'
    UnderExposed = DIR + 'UnderExposed/'
    imgSet1 = prepare_image_array(Normal)
    imgSet2 = prepare_image_array(OverExposed)
    imgSet3 = prepare_image_array(UnderExposed)

    # size of image each directory
    print(len(imgSet1.shape))
    TotalNormalImg = imgSet1.shape[0]
    TotalOverExposedImg = imgSet2.shape[0]
    TotalUnderExposedImg = imgSet3.shape[0]

    imgSet = np.concatenate((imgSet1, imgSet2, imgSet3), axis=0)
    labelSet1 = np.zeros(TotalNormalImg, dtype=np.uint8)
    labelSet2 = np.ones(TotalOverExposedImg, dtype=np.uint8)
    label_o = np.ones(TotalUnderExposedImg, dtype=np.uint8)
    labelSet3 = np.add(label_o, label_o)
    labelSet = np.concatenate((labelSet1, labelSet2, labelSet3), axis=0)

    print(labelSet)
    TotalImg = imgSet.shape[0]
    indices = np.arange(TotalImg)
    random.shuffle(indices)
    imgSet = imgSet[indices]
    labelSet = labelSet[indices]

    range = int(TotalImg * TRAIN_SPLIT)
    trainX, trainY, testX, testY = imgSet[:range], labelSet[:
                                                            range], imgSet[range:], labelSet[range:]
    return trainX, trainY, testX, testY


def prepare_image_array(imgDir):
    imgList = os.listdir(imgDir)
    imgSet = []
    for i in range(50):
        imgPath = imgDir + imgList[i]
        if(os.path.exists(imgPath)):
            img = cv2.imread(imgPath)
            resize = cv2.resize(img, (imgW, imgH))
            rgbImg = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
            imgSet.append(rgbImg)
        else:
            print("No image found.")

    imgSet = np.array(imgSet, dtype=np.uint8)
    return imgSet


if __name__ == '__main__':
    main()
