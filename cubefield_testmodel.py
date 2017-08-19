from cubefield_screen import get_img, get_img2
from cubefield_getkeys import key_check, LEFT, RIGHT, CTRL, Q, S
from cubefield_inputkeys import PressKey, ReleaseKey
import time
import os
import cv2
import numpy as np
from cubefield_alexnet import alexnet

EPOCHS  = 8
WIDTH   = 80
HEIGHT  = 60
LR      = 1e-3

MODEL_NAME = 'cubefield_ai-{}-{}-{}-{}.model'.format(LR, 'alexnet', EPOCHS)

t_time = 0.09

def forward():
    ReleaseKey(LEFT)
    ReleaseKey(RIGHT)

def left():
    PressKey(LEFT)
    ReleaseKey(RIGHT)
    time.sleep(t_time)
    ReleaseKey(LEFT)

def right():
    PressKey(RIGHT)
    ReleaseKey(LEFT)
    time.sleep(t_time)
    ReleaseKey(RIGHT)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False

    last_time = time.time()
    loops = 0
    sum_time = 0
    while(True):
            _, o_img = get_img2()


            if not paused:
                nn_image = cv2.resize(o_img, (80, 60))
                nn_image = cv2.cvtColor(nn_image, cv2.COLOR_BGR2GRAY)

                prediction = model.predict([nn_image.reshape(WIDTH,HEIGHT,1)])[0]
                moves = list(np.around(prediction))
                print(moves, prediction)

                if moves == [1, 0]:
                    left()
                elif moves == [0, 1]:
                    right()
                elif moves == [0, 0]:
                    forward()

                
            keys = key_check()

            # Suspend
            if CTRL in keys and S in keys:
                if not paused:
                    paused = True
                    print("Suspended.")
                    time.sleep(1)
                else:
                    paused = False
                    print("Resumed.")
                    ReleaseKey(LEFT)
                    ReleaseKey(RIGHT)
                    time.sleep(1)
            
            # Save and quit
            if CTRL in keys and Q in keys:
                print("Quitting")
                ReleaseKey(LEFT)
                ReleaseKey(RIGHT)
                time.sleep(1)
                break

            # FPS Calculations
            sum_time += time.time() - last_time
            last_time = time.time()

            if loops % 100 == 0:
                print('FPS avg is {} seconds'.format(1/(sum_time/100)))
                sum_time = 0
                loops = 0
            loops += 1
    
main()
