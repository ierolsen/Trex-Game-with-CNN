import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

imgs = glob.glob("data/img/*.png")

width = 125
height = 50

X = []
y = []

for img in imgs:

    filename = os.path.basename(img)
    label = filename.split("_")[0] # up, down, right
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im/255 # normalization
    X.append(im)
    y.append(label)

X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)

# visualization
#sns.countplot(y)
#plt.show()


def one_hot_labels(values):

    # Label Encoding -> One Hot Encoding

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded

# One Hote Encoding
Y = one_hot_labels(y)

# train test split
train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)

# CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="Adam",
              metrics=["acc"])

model.fit(train_X, train_y, epochs = 35, batch_size = 64)

score_train = model.evaluate(train_X, train_y)
print("Training Score: %", score_train[1]*100)

score_test = model.evaluate(test_X, test_y)
print("Test Score: %", score_test[1]*100)

# save weights
open("trex_model.json","w").write(model.to_json())
model.save_weights("trex_weight.h5")

