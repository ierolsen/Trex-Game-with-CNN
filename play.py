from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
import os
from mss import mss

print(os.getcwd())

mon = {"top":370,
       "left":700,
       "width":200,
       "height":145}

sct = mss()

width = 125
height = 50

# load model
model = model_from_json(open("trex_model.json", "r").read())
model.load_weights("trex_weight.h5")

#down:0, right:1, up:2
labels = ["Down", "Right", "Up"]

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

