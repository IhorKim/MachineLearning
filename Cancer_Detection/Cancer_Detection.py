import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import warnings
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import load_img, img_to_array

warnings.filterwarnings("ignore")

# count the number of images in the respective classes: 0-Brain Tumor and 1-Healthy
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
        print(f"{p} Folder already exists")


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
    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, rescale=1 / 255,
                                    horizontal_flip=True)  # Data Augmentation
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode="binary")
    return image


path = "train"
train_data = preprocessingImages(path)


def preprocessingImagesTest(path):
    image_data = ImageDataGenerator(rescale=1 / 255)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode="binary")
    return image


path = "test"
test_data = preprocessingImagesTest(path)
path = "val"
val_data = preprocessingImagesTest(path)

# Add early stopping and model check point
es = EarlyStopping(monitor="accuracy", min_delta=0.01, patience=3, verbose=1, mode="auto")
mc = ModelCheckpoint(monitor="accuracy", filepath="./bestmodel.h5", verbose=1, save_best_only=True, mode="auto")
cd = [es, mc]

# Model Training
history = model.fit_generator(generator=train_data, steps_per_epoch=8, epochs=30, verbose=1, validation_data=val_data,
                              validation_steps=16, callbacks=cd)

# Model plots
h = history.history
plt.plot(h["accuracy"])
plt.title("Accuracy")
plt.show()

plt.plot(h["loss"])
plt.title("Loss")
plt.show()

# Model Accuracy
model = load_model("bestmodel.h5")
acc = model.evaluate_generator(test_data)[1]
print(f"The accuracy of our model is {acc * 100}%")

# Check our model
path = "archive/Brain Tumor/Tr-gl_0023.jpg"
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


