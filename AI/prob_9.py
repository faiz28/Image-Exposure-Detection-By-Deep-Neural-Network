from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
m = 28
n = 28
c = 10


def build_model():
    inputs = Input((m, n))
    x = Flatten()(inputs)
    x = Dense(units=64, activation='sigmoid')(x)
    x = Dense(units=32, activation='sigmoid')(x)
    x = Dense(units=16, activation='sigmoid')(x)
    outputs = Dense(units=c, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def main():
    model = build_model()
    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()


if __name__ == '__main__':
    main()
