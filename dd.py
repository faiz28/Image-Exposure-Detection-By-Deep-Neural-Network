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
import csv
matplotlib.use('TkAgg')

TRAIN_SPLIT = 0.85
imgH = 32
imgW = 32
DIR = './train/'


def main():
    imgSet, labelSet = preprocess_data()

    # Train and save model.

    perameters = model_perameter()

    for perameter in perameters:

        # create path
        path = DIR + 'output/' + \
            str("loss = " + perameter[0]+", optimizer = "+perameter[1]+"/")
        if not os.path.isdir(path):
            os.mkdir(path)

        # error showing
        error_file = open(path+"error.txt", "w")

        # write accuracy into CSV
        accuracy_file_path = open(DIR+"output/accuracy.csv", "a")
        accuracy_file = csv.writer(accuracy_file_path)
        header_info = ["loss", "optimizer", "accuracy"]
        accuracy_file.writerow(header_info)

        for num in range(int(perameter[3])):
            trainX, trainY, testX, testY = suffle_data(imgSet, labelSet)
            # make directory

            newpath = path + str(num)+"/"
            if not os.path.isdir(newpath):
                os.mkdir(newpath)

            modelPath = newpath+'VGG_BinaryClassifier.hdf5'
            try:
                training(trainX, trainY, newpath, modelPath, perameter)
                accuracy = testing(testX, testY, newpath, modelPath)
                accuracy_info = [perameter[0], perameter[1], accuracy]
                accuracy_file.writerow(accuracy_info)
            except:
                error_file.write("Error occurred in loss = " +
                                 perameter[0] + " "+perameter[1] + " number = " + str(num))

        error_file.close()


def suffle_data(imgSet, labelSet):
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


def model_perameter():
    file = open('model_perameter.csv')
    csvreader = csv.reader(file)
    rows = []
    count = 0
    for row in csvreader:
        if count == 0:
            count += 1
            continue
        rows.append(row)
    print(rows)
    return rows


def testing(testX, testY, path, modelPath):
    # Preprocess image data to be fit with VGG16
    testXX = preprocess_input(testX)

    # Load the trained model
    model = load_model(modelPath)

    # Predict Class
    predictions = model.predict(testX)
    print(predictions)
    f = open(path+"predictions.txt", "w")
    f.write(str(predictions))
    f.close()

    print(testY)
    f = open(path+"acutalOutput.txt", "w")
    f.write(str(testY))
    f.close()

    predictedClass = []
    actualClass = []
    for a in predictions:
        a = np.argmax(a)
        if a == 0:
            predictedClass.append("Normal")
        elif a == 1:
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
    display_images_with_predictions(testX, predictedClass, path)

    # Evaluate model performance
    model.compile(metrics='accuracy')
    loss, accuracy = model.evaluate(testXX, testY)
    print('Accuracy: {}'.format(accuracy))

    return accuracy


def training(trainX, trainY, path, modelPath, perameter):
    trainX = preprocess_input(trainX)  # normalization data

    # Build model architecture.
    model = build_model()

    # Train model
    model.compile(loss=perameter[0], optimizer=perameter[1])
    callbackList = [EarlyStopping(monitor='val_loss', patience=20), History()]
    history = model.fit(
        trainX, trainY, epochs=int(perameter[2]), validation_split=0.2, callbacks=callbackList)

    # Save trained model and figure of training and validation loss.
    model.save(modelPath)
    figPath = path + 'Training_Vs_Val_Loss-0.png'
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


def display_images_with_predictions(imgSet, labelSet, path):
    plt.figure(figsize=(20, 20))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.title(labelSet[i])
        plt.imshow(imgSet[i])
        plt.axis('off')
    # plt.show()
    plt.legend()

    plt.savefig(path+'data.png')


def preprocess_data():
    # Load image data
    imgDir = DIR + 'Normal/'
    imgSet1 = prepare_image_array(imgDir)
    m = imgSet1.shape[0]

    imgDir = DIR + 'OverExposed/'
    imgSet2 = prepare_image_array(imgDir)
    n = imgSet2.shape[0]

    imgDir = DIR + 'UnderExposed/'
    imgSet3 = prepare_image_array(imgDir)
    o = imgSet3.shape[0]

    # Put all image data into one array.
    imgSet = np.concatenate((imgSet1, imgSet2, imgSet3), axis=0)
    print(imgSet.shape)

    # Prepare labels.
    labelSet1 = np.zeros(m, dtype=np.uint8)
    labelSet2 = np.ones(n, dtype=np.uint8)
    label_o = np.ones(o, dtype=np.uint8)
    labelSet3 = np.add(label_o, label_o)
    labelSet = np.concatenate((labelSet1, labelSet2, labelSet3), axis=0)
    print(labelSet)

    return imgSet, labelSet


def prepare_image_array(imgDir):
    imgList = os.listdir(imgDir)
    # print(imgList)
    n = len(imgList)

    imgSet = []
    for i in range(100):
        imgPath = imgDir + imgList[i]
        if (os.path.exists(imgPath)):
            img = cv2.imread(imgPath)
            resizedImg = cv2.resize(img, (imgW, imgH))
            rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
            imgSet.append(rgbImg)
        else:
            print("It is not a valid image path.")

    imgSet = np.array(imgSet, dtype=np.uint8)

    return imgSet


def build_model():
    baseModel = VGG16(input_shape=(imgH, imgW, 3), include_top=False)
    baseModel.summary()

    for layer in baseModel.layers:
        layer.trainable = False
    baseModel.summary()

    inputs = baseModel.input
    x = baseModel.output
    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dense(units=4096, activation='relu')(x)
    outputs = Dense(units=2, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.summary()

    return model


if __name__ == '__main__':
    main()
