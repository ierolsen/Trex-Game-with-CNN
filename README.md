# Trex Game with CNN

In this repo, I created a model that play Trex Game. I need to explain that, this model is not a Reinforcement Learning (RL) model. It is a simple CNN model which predict objects and keyboard actions. 

### Content:
##### 1) Getting Data
##### 2) Train a CNN Model
##### 3) Test the Trained Model in Game


---

# Getting Data

First I started with getting data. In order to do that, I used some libraries;
```python
import keyboard
import uuid
import time
from PIL import Image
from mss import mss
```

Using the library called **keyboard** I saved my action for example "up","down" and "rigt".
To import that, first the library must be installed:
```
pip install keyboard
```
Using this library called **uuid**, I can record my screen for game. When I run the all lines, I'll switch game screen and the model that is trained with data I got, will predict action according to images.

Also the library called **mss** helps me to cut off some area in the screen. Thus, model can only focuses determined area. But at the first, this librariy must be installed like others. For that:
```
pip install mss
```

After import libraries, I will set coordinates of game, because in the screen there are a lot of useless stuff, in order to remove them I will determine some coordinates using Paint :) 

![size](https://user-images.githubusercontent.com/30235603/101773976-c9aab480-3aed-11eb-9722-0e73f10bc406.png)

```python
mon = {"top":370,
       "left":700,
       "width":200,
       "height":145}
```
Using the library called **mss**, I can easly cut off the area that I want model to see only. Therefore I defined **mss**
```python
sct = mss()
```
Now, I will create a function that I will use for recording.

```python
i = 0
def record_screen(record_id, key):
    global i # I will use this i inside and outside of function. (I have an other i)

    i += 1
    print(f"{key}, {i}") #key: char of keyboard, i: num of press for char
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save(f"data/img/{key}_{record_id}_{i}.png")
```
After this record function, I need to define an other function for exit. When I want to exit in recording, I will press "esc" and then exit function will be called.
For that:
```python
is_exit = False

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey("esc", exit)
```
After all, I can set last stuff. Here I will define 
```python
record_id = uuid.uuid4()
while True:

    if is_exit: break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id=record_id, key="up")
            time.sleep(0.1)

        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id=record_id, key="down")
            time.sleep(0.1)

        elif keyboard.is_pressed("right"):
            record_screen(record_id=record_id, key="right")
            time.sleep(0.1)

    except RuntimeError: continue
```
Now, I am ready to get data in game. 
![get_data](https://user-images.githubusercontent.com/30235603/101773975-c9121e00-3aed-11eb-997a-801a0e9fdad8.png)

---

# 2) Train a CNN Model
In this content, I will train a CNN model using the data I got. The model will predict **keyboard actions** according to images. For example if model recognises a cactus, the action will be **"UP"**, or if model recognises a bird, the action will be **"DOWN"** or **"UP"**. In order to take place that, first I start with that import libraries.
```python
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
```
after libraries, I'll define path of images:
```python
imgs = glob.glob("data/img/*.png")
```
I'll resize my images
```python
width = 125
height = 50
```
Before training, I need to apply some operations like resize, normalization.
Here our labels are **keyboard actions** like **"UP"**, **"DOWN"** **"RIGHT"**. To reach this actions:
First of all I need to find file names, this line gives me that:
```python
filename = os.path.basename(img)
```
>> **OUTPUT:** "down_022f78bc-435f-4978-8524-ff1ea1a40d9a_1.png"

Then I'll use split method and seperate them as "_", after that the first index will be our label. 
```python
label = filename.split("_")[0]
```
>> **OUTPUT:** "down"

Also you can check the **record_screen function** to how save the images in first section.
After determined labels, I'll resize and normalization images
```python
im = np.array(Image.open(img).convert("L").resize((width, height)))
```
Here is the all collectively steps what I told above;
```python
X = [] # images ("cactus", "bird")
y = [] # labels ("up", "right", "down")

for img in imgs:

    filename = os.path.basename(img)
    label = filename.split("_")[0] # up, down, right
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im/255 # normalization
    X.append(im)
    y.append(label)
```
For slipt data as train and test, I must convert them to array
```python
X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)
```
Now, I will apply firstly **Label Encoding** and then **One Hot Encoding** to Y (labels).
Thus, firstly labels will be numeric for example;
**UP --> 0**
**DOWN --> 1** 
**RIGHT --> 2**
After **Label Encoding** I will apply **One Hot Encoding**
Thus, labels will be Binary Value, for example,
**0 --> 000**
**1 --> 010**
**2 -->001**
In order to do that, I define a function:
```python
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
```
Using X and Y, I will **split** my data in 0.25 rate, the part of %75 will be train, other %25 will be test size. 
```python
# train test split
train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)
```
After Split Data, I will create **Convolutional Neural Network**
```python
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

# training            
model.fit(train_X, train_y, epochs = 35, batch_size = 64)              
```
After training, I will evaluate Train and Test score
```python
score_train = model.evaluate(train_X, train_y)
print("Training Score: %", score_train[1]*100)

score_test = model.evaluate(test_X, test_y)
print("Test Score: %", score_test[1]*100)
```
![training](https://user-images.githubusercontent.com/30235603/101773978-ca434b00-3aed-11eb-8138-c38199f9bc4c.png)
To use trained model in the game, I need to save it.
```python
# save weights
open("trex_model.json","w").write(model.to_json())
model.save_weights("trex_weight.h5")
```
and finally I have a Trained CNN Model. 

---

# 3) Test the Trained Model in Game
In this section, I will test my model in the game as real-time.

For that, first I will import my libraries

```python
from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
import os
from mss import mss
```
For screen, I'll set up my sizes
```python
mon = {"top":370,
       "left":700,
       "width":200,
       "height":145}
       
sct = mss()

# size of images
width = 125
height = 50
```
Now, I can upload my Trained CNN Model
```python
# load model
model = model_from_json(open("trex_model.json", "r").read())
model.load_weights("trex_weight.h5")
```
I will determine my labels again because the model predicts images, I make it choose in this list according to output of the model. I do not touch keyboard, model does.
```python
#down:0, right:1, up:2
labels = ["Down", "Right", "Up"]
```
 
```python
framerate_time = time.time()
counter = 0
i = 0
delay = 0.4
key_down_pressed = False
while True:

    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255  # normalization

    X = np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1)
    r = model.predict(X)

    result = np.argmax(r)

    if result == 0: #down: 0
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True

    elif result == 2: #up: 2

        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
            time.sleep(delay)

        keyboard.press(keyboard.KEY_UP)

        if i < 1500:
            time.sleep(0.3)

        elif 1500 < i and i < 5000:
            time.sleep(0.2)

        else:
            time.sleep(0.17)

        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)

    counter += 1

    if (time.time() - framerate_time) > 1:

        counter = 0
        framerate_time = time.time()

        if i <= 1500:
            delay -= 0.003

        else:
            delay -= 0.005

        if delay < 0:
            delay = 0

        print("----------------")
        print(f"Down: {r[0][0]}\nRight: {r[0][1]}\nUp: {r[0][2]}")

        i += 1
```
Now, I need to explain what this while loop does.

First, I got an area according to the pixel (mon), and then I converted it. After Converting, I applied **resize** and **normalization**
```python
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255  # normalization
```
After these steps, I turned it array and reshaped it. Because the input is known for model.
The **np.argmax()** returns the indices of the maximum values along an axis. I use it due to find the label maximum probability.
```python
    X = np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1)
    r = model.predict(X)

    result = np.argmax(r)
```
If the result is 0 it means **DOWN**, using **keyboard library**, model will press **DOWN**.
And after that, I changed it as **key_down_pressed = True**
```python
    if result == 0: #down: 0
    keyboard.press(keyboard.KEY_DOWN)
    key_down_pressed = True
```
If the result is 2 it means **UP**, again using **keyboard library** model will press **UP**
Here, **i<1500** is random number. (1500 is frame)
```python
    elif result == 2: #up: 2

        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
            time.sleep(delay)

        keyboard.press(keyboard.KEY_UP)

        if i < 1500:
            time.sleep(0.3)

        elif 1500 < i and i < 5000:
            time.sleep(0.2)

        else:
            time.sleep(0.17)
```
After that, the dinosaur downs again but I must **release** it because, if I don't, dinosaur keep to stay down.
```python
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
```
Now here, I set up **delay**
```python
    counter += 1

    if (time.time() - framerate_time) > 1:

        counter = 0
        framerate_time = time.time()

        if i <= 1500:
            delay -= 0.003 # 3 msec

        else:
            delay -= 0.005 # 5 msec

        if delay < 0:
            delay = 0

        print("----------------")
        print(f"Down: {r[0][0]}\nRight: {r[0][1]}\nUp: {r[0][2]}")

        i += 1
```
![trex_gif](https://s8.gifyu.com/images/trex.gif)

