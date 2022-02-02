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

TRAIN_SPLIT = 0.85

DIR = './train/'


def main():
    # original_model_prediction()

    # Prepare data sets
    imgH = 32
    imgW = 32
    trainX, trainY, testX, testY = preprocess_data(imgH, imgW)

    # Train and save model.
    modelPath = DIR + 'VGG_BinaryClassifier.hdf5'
    training(trainX, trainY, modelPath, imgH, imgW)

    # Load and test the model
    testing(testX, testY, modelPath)


def testing(testX, testY, modelPath):
    # Preprocess image data to be fit with VGG16
    testXX = preprocess_input(testX)

    # Load the trained model'
    model = load_model(modelPath)

    # Predict Class
    predictions = model.predict(testX)
    print("predictions : ")
    print(predictions)
    predictedClass = []
    actualClass = []
    for a in predictions:
        if a < 0.33:
            predictedClass.append("Normal")
        elif a < 0.66:
            predictedClass.append("OverExposed")
        else:
            predictedClass.append("UnderExposed")
    for a in testY:
        if a < 0.33:
            actualClass.append("Normal")
        elif a < 0.66:
            actualClass.append("OverExposed")
        else:
            actualClass.append("UnderExposed")
    # predictedClass = ['Normal' if a < 0.33 elif a < 0.66 'WaterLilly' else 'WaterLilly' for a in predictions]
    # actualClass = ['Jasmine' if a < 0.5 else 'WaterLilly' for a in testY]
    print('Actual Class: {}'.format(actualClass))
    print('Predicted Class: {}'.format(predictedClass))
    display_images_with_predictions(testX, predictedClass)

    # Evaluate model performance
    model.compile(metrics='accuracy')
    loss, accuracy = model.evaluate(testXX, testY)
    print('Accuracy: {}'.format(accuracy))


def training(trainX, trainY, modelPath, imgH, imgW):
    # Preprocess image data to be fit with VGG16
    print(trainX.max(), trainX.min())
    trainX = preprocess_input(trainX)  # normalization data
    print(trainX.max(), trainX.min())

    # Build model architecture.
    model = build_model(imgH, imgW)

    # Train model
    model.compile(loss='mse', optimizer='rmsprop')
    callbackList = [EarlyStopping(monitor='val_loss', patience=20), History()]
    history = model.fit(trainX, trainY, epochs=100,
                        validation_split=0.2, callbacks=callbackList)

    # Save trained model and figure of training and validation loss.
    model.save(modelPath)
    figPath = DIR + 'Training_Vs_Val_Loss.png'
    plot_loss(history, figPath)


def plot_loss(history, figPath):
    loss = history.history['loss']
    valLoss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(20, 20))
    plt.rcParams['font.size'] = '14'
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, valLoss, 'k*-', label='Validation loss')
    plt.title('Training and validation loss - 0')
    plt.legend()

    plt.savefig(figPath)
    plt.close()


def original_model_prediction():
    # Load a pre-trained model.
    model = VGG16()
    model.summary()

    # Prepare data set.
    _, _, testX, testY = preprocess_data(224, 224)

    # Predict which class the loaded image belongs to
    # List of 1000 classes: http://image-net.org/challenges/LSVRC/2014/browse-synsets
    predictions = model.predict(testX)
    predictions = decode_predictions(predictions, top=1)

    n = testX.shape[0]
    predictedClass = []
    for i in range(n):
        className = predictions[i][0][1]
        predictedClass.append(className)

    # Draw some example images with predictions
    display_images_with_predictions(testX, predictedClass)


def display_images_with_predictions(imgSet, labelSet):
    plt.figure(figsize=(20, 20))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.title(labelSet[i])
        plt.imshow(imgSet[i])
        plt.axis('off')
    # plt.show()
    plt.legend()

    plt.savefig(figPath)


def preprocess_data(imgW, imgH):
    # Load image data
    imgDir = DIR + 'Normal/'
    imgSet1 = prepare_image_array(imgDir, imgW, imgH)
    print(imgSet1.shape)
    m = imgSet1.shape[0]

    imgDir = DIR + 'OverExposed/'
    imgSet2 = prepare_image_array(imgDir, imgW, imgH)
    print(imgSet2.shape)
    n = imgSet2.shape[0]

    imgDir = DIR + 'UnderExposed/'
    imgSet3 = prepare_image_array(imgDir, imgW, imgH)
    print(imgSet3.shape)
    o = imgSet3.shape[0]

    # Put all image data into one array.
    imgSet = np.concatenate((imgSet1, imgSet2, imgSet3), axis=0)
    print(imgSet.shape)

    # Prepare labels.
    labelSet1 = np.zeros(m, dtype=np.uint8)
    labelSet2 = np.ones(n, dtype=np.uint8)
    label_o = np.zeros(o, dtype=np.uint8)
    labelSet3 = np.add(label_o, label_o)
    labelSet = np.concatenate((labelSet1, labelSet2, labelSet3), axis=0)
    print(labelSet)

    # Shuffle image data and labels
    p = imgSet.shape[0]  # p = n + m + o
    indices = np.arange(p)
    # print(indices)
    random.shuffle(indices)
    # print(indices)
    imgSet = imgSet[indices]
    labelSet = labelSet[indices]
    print("Label set : ")
    print(labelSet)

    # Split data into training and test sets
    r = int(p * TRAIN_SPLIT)
    trainX = imgSet[:r]
    trainY = labelSet[:r]
    testX = imgSet[r:]
    testY = labelSet[r:]

    return trainX, trainY, testX, testY


def prepare_image_array(imgDir, imgW, imgH):
    imgList = os.listdir(imgDir)
    # print(imgList)
    n = len(imgList)

    imgSet = []
    for i in range(n):
        imgPath = imgDir + imgList[i]
        if (os.path.exists(imgPath)):
            # print(imgPath)

            # Load image.
            img = cv2.imread(imgPath)
            # print(img.shape)

            # Resize image.
            resizedImg = cv2.resize(img, (imgW, imgH))
            # print(resizedImg.shape)

            # Convert BGR image into RGB image.
            rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)

            # Put image into a list
            imgSet.append(rgbImg)
        else:
            print("It is not a valid image path.")

    print("total image "+str(len(imgSet)))
    imgSet = np.array(imgSet, dtype=np.uint8)
    print("total shape "+str(imgSet.shape))

    return imgSet


def build_model(imgH, imgW):
    baseModel = VGG16(input_shape=(imgH, imgW, 3), include_top=False)
    baseModel.summary()

    for layer in baseModel.layers:
        layer.trainable = False
    baseModel.summary()

    inputs = baseModel.input
    x = baseModel.output
    x = Flatten()(x)
    x = Dense(8, activation='sigmoid')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.summary()

    return model


if __name__ == '__main__':
    main()
