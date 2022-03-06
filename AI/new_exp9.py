from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model

m = 28
n = 28
h = 4
c = 10
input_layer = Input((m, n))
x = Flatten()(x)
x = Dense(8, activation='softmax')(input_layer)
for i in range(h - 1):
    x = Dense(8, activation='softmax')(x)
output_layer = Dense(c)(x)
model = Model(input_layer, output_layer)
model.summary()
