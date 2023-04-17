import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import warnings
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import load_img, img_to_array

warnings.filterwarnings("ignore")

# calculate how many pictures there are in the 0-Brain Tumor and 1-Healthy categories
ROOT_DIR = "archive"
number_of_images = {}

for dir in os.listdir(ROOT_DIR):
    number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))
print(number_of_images.items())


# split the data: 70% - train data, 15% - for validation, 15% - for testing
def dataFolder(p, split):
    # create a train folder
    if not os.path.exists("./" + p):
        os.mkdir("./" + p)

        for dir in os.listdir(ROOT_DIR):
            os.makedirs("./" + p + "./" + dir)
            for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR, dir)),
                                        size=(math.floor(split * number_of_images[dir]) - 5),
                                        replace=False):
                O = os.path.join(ROOT_DIR, dir, img)
                D = os.path.join("./" + p, dir)
                shutil.copy(O, D)
                os.remove(O)
    else:
        print(f"{p} folder already exists")


dataFolder("train", 0.7)
dataFolder("val", 0.15)
dataFolder("test", 0.15)

# Model building
# CNN model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(filters=36, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(rate=0.25))
model.add(Dense(units=1, activation="sigmoid"))
model.summary()
model.compile(optimizer="adam", loss=tf.keras.losses.binary_crossentropy, metrics=["accuracy"])


# preparing data using Data Generator
def preprocessingImages(path):
    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, preprocessing_function=preprocess_input,
                                    horizontal_flip=True)  # data Augmentation
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode="binary")
    return image


path = "train"
train_data = preprocessingImages(path)


def preprocessingImagesTest(path):
    image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode="binary")
    return image


path = "test"
test_data = preprocessingImagesTest(path)
path = "val"
val_data = preprocessingImagesTest(path)

# Transfer Learning. Combine our model with pre-trained MobileNet model
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)
for layer in base_model.layers:
    layer.trainable = False

# Change the Mobile Net Model adding Flatten and Dense layers
X = Flatten()(base_model.output)
X = Dense(units=1, activation="sigmoid")(X)

model = Model(base_model.input, X)

model.compile(optimizer="rmsprop", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

mc = ModelCheckpoint(filepath="bestmodel.h5", monitor="accuracy", verbose=1, save_best_only=True)
es = EarlyStopping(monitor="accuracy", min_delta=0.01, patience=3, verbose=1)
cb = [mc, es]

# Model Training
history = model.fit_generator(generator=train_data, steps_per_epoch=8, epochs=30, validation_data=val_data,
                              validation_steps=16, callbacks=cb)

# Model Accuracy
model = load_model("bestmodel.h5")
acc = model.evaluate_generator(test_data)[1]
print(f"The accuracy of our model is {acc * 100}%")

# Model plots
h = history.history
plt.plot(h["accuracy"])
plt.title("Accuracy")
plt.show()

plt.plot(h["loss"])
plt.title("Loss")
plt.show()

# Check our model
path = "archive/Healthy/Tr-no_0044.jpg"
img = load_img(path, target_size=(224, 224))
input_arr = img_to_array(img) / 255

plt.imshow(input_arr)
plt.show()

input_arr = np.expand_dims(input_arr, axis=0)
pred = (model.predict(input_arr) > 0.5).astype("int32")
if pred == 0:
    print("The MRI is having a Tumor")
else:
    print("The MRI is Not having a Tumor")
print(train_data.class_indices)
