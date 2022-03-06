from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras import Model

m = 28
n = 28
h = 4
c = 10
input_layer = Input((m, n, 1))
x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
for i in range(h - 1):
    x = Conv2D(filters=5, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
output_layer = Dense(c)(x)
model = Model(input_layer, output_layer)
model.summary()
