import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import os

VERSION = 'shuffle'
SIZE = '40k'

shuffle_batch = False
batch_size = 10

count = 0

while os.path.isfile('cubefield_data_{}raw_{}.npy'.format(SIZE, count)):

    print("Loading {}.".format('cubefield_data_{}raw_{}.npy'.format(SIZE, count)))
    train_data = np.load('cubefield_data_{}raw_{}.npy'.format(SIZE, count))
    final_data_file = 'cubefield_data_{}_{}_{}.npy'.format(SIZE, VERSION, count)
    count += 1


    df = pd.DataFrame(train_data)
    print(len(train_data))
    print(df.head())
    print(Counter(df[1].apply(str)))

    lefts = []
    rights = []
    forwards = []

    if not shuffle_batch:
        shuffle(train_data)
        for data in train_data:
            img = data[0]
            direction = data[1]

            if direction == [1, 0]:
                lefts.append([img, direction])
            elif direction == [0, 1]:
                rights.append([img, direction])
            elif direction == [0, 0]:
                forwards.append([img, direction])
            else:
                print('wtf input did you give me?')
        
        forwards = forwards[:len(lefts)][:len(rights)]
        lefts = lefts[:len(forwards)]
        rights = rights[:len(forwards)]

        train_data = forwards + lefts + rights

        shuffle(train_data)
    else:
        train_data = train_data[:-(len(train_data)%batch_size)]
        data_len = len(train_data)
        
        train_data = [train_data[i:i+batch_size] for i in range(0, data_len, batch_size)]
        shuffle(train_data)
        train_data = np.array(train_data).reshape(data_len, 2)
        

    df = pd.DataFrame(train_data)
    print(Counter(df[1].apply(str)))

    np.save(final_data_file, train_data)

##    
##    cv2.imshow('test', img)
##    print(choice)
##    if cv2.waitKey(25) & 0xFF == ord('q'):
##        cv2.destroyAllWindows()
##        break
##    
