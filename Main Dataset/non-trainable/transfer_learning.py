# Faiz ahmed
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, History
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import SGD, Adam,schedules
import random
import os,load_image
import cv2
import matplotlib
import matplotlib.pyplot as plt

DIR = './'
TRAIN_SPLIT = 0.85
imgH = 32
imgW = 32

file = open(DIR+'tf.txt',"w")
def main():
    trainX, trainY, testX, testY =  load_image.load_image_data()
    modelPath = DIR + 'VGG_Classifier.hdf5'
    training(trainX, trainY, modelPath)
    testing(testX, testY, modelPath)


def testing(testX, testY, modelPath):
    model = load_model(modelPath)
    model.compile(metrics='accuracy')
    loss, accuracy = model.evaluate(testX, testY)
    print('Accuracy: {}'.format(accuracy))
    
    file.write('Accuracy: {}'.format(accuracy))
    
     # making confusion marix
    predicted_output = []
    prediction = model.predict(testX)
    for i in prediction:
        predicted_output.append(np.argmax(i)) 

    actual_output = []
    for i in testY:
        actual_output.append(np.argmax(i)) 
    con_mat = confusion_matrix(actual_output, predicted_output)
    for i in con_mat:
        print(i)
        file.write(str(i)+"\n")


    
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)



  # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

sgd = SGD(learning_rate=lr_schedule)

def training(trainX, trainY, modelPath):
    model = build_model()
    model.compile(loss='mse', optimizer=sgd,metrics  = ['accuracy'])
    callbackList = [EarlyStopping(monitor='val_loss', patience=50), History()]
    history = model.fit(trainX, trainY, epochs=500, batch_size=128, validation_split=0.2, callbacks=callbackList)
    model.save(modelPath)
    plot_loss_acc(history)


def plot_loss_acc(history):
    accuracy = history.history['accuracy']
    valaccuracy = history.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    plt.figure(figsize=(20, 20))
    plt.rcParams['font.size'] = '14'
    plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
    plt.plot(epochs, valaccuracy, 'k*-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(DIR + 'training_validation_accuracy.png')
    plt.close()
    
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
    # x = Dense(units=1024, activation="relu")(x)
    # x = Dense(units=512, activation="relu")(x)
    x = Dense(units=256, activation="sigmoid")(x)
    x = Dropout(.3)(x)
    # x = Dense(units=128, activation="relu")(x)
    x = Dense(units=64, activation="sigmoid")(x)
    # x = Dense(units=32, activation="relu")(x)
    # x = Dense(units=8, activation="sigmoid")(x)
    outputs = Dense(units=3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.summary()

    return model


if __name__ == '__main__':
    main()
