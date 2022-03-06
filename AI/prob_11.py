from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
m = 28
n = 28
c = 2


def build_model():
    inputs = Input((m, n, 1))
    x = Conv2D(filters=5, kernel_size=(2, 2),
               strides=(1, 1), padding='same')(inputs)
    x = Conv2D(filters=5, kernel_size=(2, 2),
               strides=(1, 1), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(units=16, activation='sigmoid')(x)
    outputs = Dense(units=c, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


def prepare_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()

    # select 1,2 from mnist dataset
    train_indices = np.argwhere(np.logical_and(trainY >= 3, trainY <= 4))
    test_indices = np.argwhere(np.logical_and(testY >= 3, testY <= 4))

    trainX = trainX[train_indices.flatten()]
    trainY = trainY[train_indices.flatten()]
    testX = testX[test_indices.flatten()]
    testY = testY[test_indices.flatten()]

    trainY = to_categorical(trainY == 3)
    testY = to_categorical(testY == 3)

    trainX = trainX.astype(np.float32)
    testX = testX.astype(np.float32)
    trainX /= 255
    testX /= 255
    print(trainX.shape)
    return trainX, trainY, testX, testY


def main():
    trainX, trainY, testX, testY = prepare_data()
    model = build_model()
    model.summary()

    model.fit(trainX, trainY, epochs=5, batch_size=1000, validation_split=.3)
    score, acc = model.evaluate(testX, testY)
    print("loss : %s, accuracy : %s\n" % (score, acc))


if __name__ == '__main__':
    main()
