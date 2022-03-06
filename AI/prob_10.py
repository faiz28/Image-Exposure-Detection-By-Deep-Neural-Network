from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
m = 28
n = 28
c = 10


def build_model():
    inputs = Input((m, n, 1))
    x = Conv2D(filters=5, kernel_size=(2, 2),
               strides=(1, 1), padding='same')(inputs)
    x = Conv2D(filters=5, kernel_size=(2, 2),
               strides=(1, 1), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(units=64, activation='sigmoid')(x)
    outputs = Dense(units=c, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='mse', optimizer='rmsprop', metrices='accuracy')
    return model


def prepare_data():
    trainX, trainY, testX, testY = mnist.load_data()

    trainY = to_categorical(trainY, c)
    testY = to_categorical(testY, c)

    print(testY)
    trainX /= 255
    testX /= 255

    return trainX, trainY, testX, testY


def main():
    trainX, trainY, testX, testY = prepare_data()
    model = build_model()
    model.summary()

    model.fit(trainX, trainY, epochs=50, validation_split=.3)

    loss, accuracy = model.evaluate(testX, testY)
    print("loss : %s, accuracy : %s\n" % (loss, accuracy))


if __name__ == '__main__':
    main()
