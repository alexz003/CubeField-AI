import numpy as np
from cubefield_alexnet import alexnet, alexnet2, alexnet3, model_lstm
import os

VERSION = 'shuffle'
SIZE = '40k'
MODEL_TYPE = 'Alexnet3'
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'cubefield_ai_{}_{}_{}_{}.model'.format(MODEL_TYPE, VERSION, SIZE, LR, EPOCHS)

LOAD_MODEL = False


count = 0
file_name = 'cubefield_data_{}_{}_{}.npy'.format(SIZE, VERSION, count)
while os.path.isfile(file_name):
    count += 1
    file_name = 'cubefield_data_{}_{}_{}.npy'.format(SIZE, VERSION, count)


model = alexnet3(WIDTH, HEIGHT, LR)

if LOAD_MODEL:
    model.load(MODEL_NAME)
    
for epoch in range(EPOCHS):
    for file_index in range(count):
        file_name = 'cubefield_data_{}_{}_{}.npy'.format(SIZE, VERSION, file_index)

        train_data = np.load(file_name)

        print("Loaded {} datapoints, prepping data.".format(len(train_data)))

        train = train_data[:-300]
        test = train_data[-300:]

        train_data = []


        frames = 10

        # Training inputs
        #X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT)
        #X = [X[i:i+frames] for i in range(0, len(X), frames)]
        # Training outputs
        #Y = [i[1] for i in train]
        #Y = [Y[i:i+frames] for i in range(0, len(Y), frames)]


        # Testing inputs
        #x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT)
        #x = [x[i:i+frames] for i in range(0, len(x), frames)]
        # Testing outputs
        #y = [i[1] for i in test]
        #y = [y[i:i+frames] for i in range(0, len(y), frames)]

        # Training inputs
        X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
        # Training outputs
        Y = [i[1] for i in train]


        train = []

        # Testing inputs
        x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
        # Testing outputs
        y = [i[1] for i in train]


        test = []

        print("Fitting model... shape={}".format(X.shape))

        model.fit({'input' : X}, {'targets' : Y}, n_epoch = 1, validation_set =
                  ({'input' : x}, {'targets' : y}), snapshot_step = 300,
                  show_metric =True, run_id = MODEL_NAME)

    model.save(MODEL_NAME)

# tensorboard --logdir=foo:D:\workspace\PythonPlays\CubeField

