from cubefield_screen import get_img
from cubefield_getkeys import key_check, LEFT, RIGHT, CTRL, Q, S
from cubefield_inputkeys import PressKey, ReleaseKey
from cubefield_alexnet import alexnet, alexnet2, alexnet3, model_lstm
import numpy as np
import time
import sys
import os
import cv2

# Train = True, Test = False
train_or_test = False
if '--test' in sys.argv:
    train_or_test = False
elif '--train' in sys.argv:
    train_or_test = True
else:
    print("Must contain --test or --train in command argument")
    quit()


VERSION = 'shuffle'
SIZE = '40k'
MODEL_TYPE = 'Alexnet3'

EPOCHS  = 8
WIDTH   = 160
HEIGHT  = 120
LR      = 1e-3

MODEL_NAME = 'cubefield_ai_{}_{}_{}_{}.model'.format(MODEL_TYPE, VERSION, SIZE, LR, EPOCHS)

turn_threshold = 0.8
t_time = 0.01

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

def keys_to_output(keys):
    output = [0, 0]

    if LEFT in keys and RIGHT in keys:
        output = [0, 0]
    elif LEFT in keys:
        output[0] = 1
    elif RIGHT in keys:
        output[1] = 1

    return output

def combine_training_data(data_iter):
    print("Combining training data (This could take a few moments)")
    if(data_iter > 0):
        fn1 = 'cubefield_data_{}.npy'.format(0)
        t_data1 = list(np.load(fn1))

        for i in range(data_iter):
            fn2 = 'cubefield_data_{}.npy'.format(i+1)
            t_data1 += list(np.load(fn2))

        print("Saving {} datapoints ".format(len(t_data1)))
        np.save('cubefield_data_0.npy', t_data1)
        

def main():

    file_name = 'cubefield_data_0.npy'
    training_data = []
    data_size = 0
    model = None

    # Load preliminary data
    if train_or_test:
        if os.path.isfile(file_name):
            print('File exists, loading previous data.')
            training_data = list(np.load(file_name))
            data_size = len(training_data)
        else:
            print('File does not exist, starting fresh.')
            gather_traindata = True
            training_data = []
    else:

        model = alexnet3(WIDTH, HEIGHT, LR)
        model.load(MODEL_NAME)
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False

    last_time = time.time()
    loops = 0
    sum_time = 0
    data_iter = 0
    while(True):
        
            keys = key_check()

            # Suspend
            if CTRL in keys and S in keys:
                if train_or_test:
                    if not paused:
                        print("Trimming and rewriting file")
                        del(training_data[-150:])
                        print(len(training_data))
                        np.save(file_name, training_data)
                        paused = True
                        print("Suspended.")
                    else:
                        paused = False
                        print("Resumed.")
                else:
                    if not paused:
                        paused = True
                        print("Suspended.")
                    else:
                        paused = False
                        print("Resumed.")
                        ReleaseKey(LEFT)
                        ReleaseKey(RIGHT)

                time.sleep(1)
            
            # Save and quit
            if CTRL in keys and Q in keys:
                if train_or_test:
                    print("Trimming and rewriting file")
                    del(training_data[-150:])
                    print(len(training_data))
                    np.save(file_name, training_data)
                    
                print("Quitting")
                break
            
            if not paused:
                _, o_img = get_img()
                    
                nn_image = cv2.resize(o_img, (WIDTH, HEIGHT))
                nn_image = cv2.cvtColor(nn_image, cv2.COLOR_BGR2GRAY)


                # Gather training data (image, inputs)
                if train_or_test:
                    output = keys_to_output(keys)

                    training_data.append([nn_image, output])

                    if len(training_data) % 1000 == 0:
                        print("Quick-saving {} datapoints".format(len(training_data)))
                        np.save(file_name, training_data)

                    if len(training_data) >= 20000:
                        print("Saving {} datapoints and switching to save file {}.".format(len(training_data), data_iter))
                        data_size += len(training_data)
                        data_iter += 1
                        file_name = 'cubefield_data_{}.npy'.format(data_iter)
                        training_data = []

                # Test modeled data
                if not train_or_test:
                    prediction = model.predict([nn_image.reshape(WIDTH,HEIGHT, 1)])[0]
                    moves = list(np.around(prediction))
                    if moves == [1, 0]:
                        if prediction[0] < turn_threshold:
                            left()
                        else:
                            forward()
                    elif moves == [0, 1]:
                        if prediction[1] < turn_threshold:
                            right()
                        else:
                            forward()
                    elif moves == [0, 0]:
                        forward()
                    


            # FPS Calculations
            sum_time += time.time() - last_time
            last_time = time.time()

            if loops % 100 == 0 and sum_time > 0:
                print('FPS avg is {} seconds'.format(1/(max(sum_time, 1e-6)/100)))
                sum_time = 0
                loops = 0
            loops += 1

    training_data = []
    if train_or_test:
        combine_training_data(data_iter)
    
main()
