from tensorflow.keras.layers import Dense, Flatten, Dropout,Input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,MaxPooling2D,AveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam,schedules
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, History
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import os,cv2
import matplotlib
import load_image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import io
imgH = 32
imgW = 32


DIR = './'
modelpath = DIR + 'Large_CNN_Classifier.hdf5'
file = open(DIR+'large_size_cnn.txt',"w")


# load data
x_train, y_train, x_test, y_test = load_image.load_image_data()




inputs = Input(shape=(imgH, imgW, 3))
x = Conv2D(filters =64, kernel_size = (3,3),activation='relu', padding="same")(inputs)
x = Conv2D(filters =64, kernel_size = (3,3),activation='relu', padding="same")(x)
x = MaxPooling2D( pool_size=(2, 2), strides=(2,2))(x)
x = Conv2D(128, (3,3), activation='relu', padding="same")(x)
x = Conv2D(128, (3,3), activation='relu', padding="same")(x)
x = MaxPooling2D( pool_size=(2, 2), strides=None, padding="same")(x)
x = Conv2D(256, (3,3), activation='relu', padding="same")(x)
x = MaxPooling2D( pool_size=(2, 2), strides=None, padding="same")(x)

# baseModel.summary()
x = Flatten()(x)
x = Dense(units=256, activation="sigmoid")(x)
x = Dropout(.3)(x)
x = Dense(units=32, activation="sigmoid")(x)
# x = Dense(units=64, activation="relu")(x)
# x = Dense(units=8, activation="sigmoid")(x)
outputs = Dense(units=3, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()
model.compile(optimizer="rmsprop",loss='mse',metrics  = ['accuracy'])



import tensorflow as tf
output = []
history_dic = []
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)



  # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

sgd = SGD(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
callbackList = [EarlyStopping(monitor='val_loss', patience=50), History()]
history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=callbackList)
loss, acc = model.evaluate(x_test, y_test)
print("\n\naccuracy  = %s"%( str(acc)))
file.write("\n\naccuracy  = %s"%( str(acc)))
model.save(modelpath)

loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: {}'.format(accuracy))
file.write('Accuracy: {}'.format(accuracy))
    
predicted_output = []
prediction = model.predict(x_test)
for i in prediction:
    predicted_output.append(np.argmax(i)) 

actual_output = []
for i in y_test:
    actual_output.append(np.argmax(i)) 
con_mat = confusion_matrix(actual_output, predicted_output)
for i in con_mat:
    print(i)
    file.write(str(i)+"\n")

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
    plt.savefig(DIR + 'Large_CNN_training_validation_accuracy.png')
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

    plt.savefig(DIR + 'Large_CNN_training_validation_loss.png')
    plt.close()
    
plot_loss_acc(history)