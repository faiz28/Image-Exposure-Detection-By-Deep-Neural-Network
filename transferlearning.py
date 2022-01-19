import numpy as np
import cv2

import PIL.Image as Image
import os

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("hello world")
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])
gold_fish = Image.open("goldfish.png").resize(IMAGE_SHAPE)
gold_fish = np.array(gold_fish)/255.0
print(gold_fish.shape)
result = classifier.predict(gold_fish[np.newaxis, ...])
print(result.shape)
predicted_label_index = np.argmax(result)
print(predicted_label_index)


import pathlib
data_dir = pathlib.Path().resolve()
print(data_dir)
flowers_images_dict = {
    'normal': list(data_dir.glob('train/Normal/*')),
    'overexposed': list(data_dir.glob('train/OverExposed/*')),
    'underexposed': list(data_dir.glob('train/UnderExposed/*')),
}
flowers_labels_dict = {
    'normal': 0,
    'overexposed': 1,
    'underexposed': 2,
}
print(flowers_images_dict['normal'][:5])

X, y = [], []
count =0

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        print(count)
        count +=1
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])
        if count>=600:
            count=0
            break

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape)
print(X_test.shape)
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

num_of_flowers = 3

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(num_of_flowers)
])

model.summary()

model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=15)

model.evaluate(X_test_scaled,y_test)

