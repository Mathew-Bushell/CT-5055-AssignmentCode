import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

imgHeight = 180
imgWidth = 180
batchSize = 32
#Sets up datasets for training and validation
trainDataDir = tf.keras.utils.image_dataset_from_directory(
    "Subset_of_posted_Items/train",
    # validation_split=0.2,
    # subset="training",
    seed=123,
    image_size=(imgHeight, imgWidth),
    batch_size=batchSize)
classNames = trainDataDir.class_names
validationDataDir = tf.keras.utils.image_dataset_from_directory(
    "Subset_of_posted_Items/validation",
    # validation_split=0.2,
    # subset="training",
    seed=123,
    image_size=(imgHeight, imgWidth),
    batch_size=batchSize)



# plt.figure(figsize=(10, 10))
# for images, labels in trainDataDir.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(classNames[labels[i]])
#     plt.axis("off")
# plt.show()