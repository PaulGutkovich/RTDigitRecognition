import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler

plt.set_cmap("binary")

train = pd.read_csv("train.csv")
y = train["label"].values.reshape(-1, 1)
X = train.drop(["label"], axis=1).values.reshape(-1, 28, 28, 1)/255.

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X[i].reshape(28, 28))
    plt.title(y[i])

plt.show()

y = to_categorical(y)

datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.10, width_shift_range=0.1, height_shift_range=0.1)

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

lr = LearningRateScheduler(lambda x: 0.01*0.95**x)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), epochs=30,
                              steps_per_epoch=X_train.shape[0]//64, validation_data=(X_val, y_val),
                              callbacks=[lr], verbose=1)

model.save("weights.h5")