import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

imgHeight = 180
imgWidth = 180
batchSize = 32
#Sets up datasets for training and validation
trainDataSet = tf.keras.utils.image_dataset_from_directory(
    "Subset_of_posted_Items/train",
    # validation_split=0.2,
    # subset="training",
    seed=123,
    image_size=(imgHeight, imgWidth),
    batch_size=batchSize)
classNames = trainDataSet.class_names
validationDataSet = tf.keras.utils.image_dataset_from_directory(
    "Subset_of_posted_Items/validation",
    # validation_split=0.2,
    # subset="training",
    seed=123,
    image_size=(imgHeight, imgWidth),
    batch_size=batchSize)


#visualises the data within the dataset (training)
# plt.figure(figsize=(10, 10))
# for images, labels in trainDataSet.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(classNames[labels[i]])
#     plt.axis("off")
# plt.show()

# the dataset is configured to increase performance while training
AUTOTUNE = tf.data.AUTOTUNE
trainDataSet = trainDataSet.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validationDataSet = validationDataSet.cache().prefetch(buffer_size=AUTOTUNE)

normalizationLayer = layers.Rescaling(1./255)
normalizedDataSet = trainDataSet.map(lambda x, y: (normalizationLayer(x), y))
imageBatch, labelsBatch = next(iter(normalizedDataSet))

#creates the model
classNum = len(classNames)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(imgHeight, imgWidth, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(classNum)
])
# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model is trained on the previously set up datasets
epochs=10
history = model.fit(
  trainDataSet,
  validation_data=validationDataSet,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


while True:
    # try:
    #     filePath = input("Paste the file path of the image you want to test here: ")
    #     if os.path.exists(filePath):
    #         img = tf.keras.utils.load_img(filePath, target_size = (imgHeight, imgWidth))
    #         imgArray = tf.keras.utils.img_to_array(img)
    #         imgArray = tf.expand_dims(imgArray, 0)#creates a batch
    #
    #         predictions = model.predict(imgArray)
    #         score = tf.nn.softmax(predictions[0])
    #
    #         print("This image most likely belongs to " + classNames[np.argmax(score)]+" with a "+(100 * np.max(score))+" percent confidence.")
    #     elif filePath == "Q":
    #         break
    #     else:
    #         print("This file path does not exist")
    # except:
    #     print("invalid input")
    filePath = input("Paste the file path of the image you want to test here: ")
    if os.path.exists(filePath):
        img = tf.keras.utils.load_img(filePath, target_size = (imgHeight, imgWidth))
        imgArray = tf.keras.utils.img_to_array(img)
        imgArray = tf.expand_dims(imgArray, 0)#creates a batch

        predictions = model.predict(imgArray)
        score = tf.nn.softmax(predictions[0])
        confidenceScore = (100 * np.max(score))
        print("This image most likely belongs to " + classNames[np.argmax(score)]+" with a "+ str(confidenceScore) +" percent confidence.")
    elif filePath == "Q":
        break
    else:
        print("This file path does not exist")

