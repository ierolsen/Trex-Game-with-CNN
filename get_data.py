import keyboard
import uuid
import time
from PIL import Image
from mss import mss

"""
http://www.trex-game.skipser.com/
"""

mon = {"top":370,
       "left":700,
       "width":200,
       "height":145}

sct = mss()

i = 0

def record_screen(record_id, key):
    global i

    i += 1
    print(f"{key}, {i}") #key: char of keyboard, i: num of press for char
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save(f"data/img/{key}_{record_id}_{i}.png")

is_exit = False

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey("esc", exit)

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