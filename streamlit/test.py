import os

import tensorflow as tf
import cv2
import numpy as np

images = []
image_names = []
for img in os.listdir('data_test'):
    image = cv2.imread(f'data_test/{img}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    
    images.append(image)
    image_names.append(f'data_test/{img}')

img_tensor = tf.convert_to_tensor(images, dtype=tf.float32)

print(img_tensor.shape)
print(image_names)

loaded = tf.saved_model.load('googlenet_model/original_data')

print(list(loaded.signatures.keys()))

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

prediction = infer(img_tensor)

print(prediction)

for i in range(3):
    print(tf.math.argmax(prediction[f'output_{i}'], axis=1))
