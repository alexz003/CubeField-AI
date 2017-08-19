#Citation: Box of Hats (https://github.com/Box-Of-Hats)

import win32api as wapi
import time

LEFT    = 0x25
RIGHT   = 0x27
CTRL    = 0x11
Q       = 0x51
S       = 0x53

def key_check():
    keys = []
    if wapi.GetAsyncKeyState(LEFT):
        keys.append(LEFT)
    elif wapi.GetAsyncKeyState(RIGHT):
        keys.append(RIGHT)

    if wapi.GetAsyncKeyState(CTRL):
        keys.append(CTRL)
    if wapi.GetAsyncKeyState(Q):
        keys.append(Q)
    if wapi.GetAsyncKeyState(S):
        keys.append(S)

    return keys

if __name__ == '__main__':
    while(True):
        if wapi.GetAsyncKeyState(LEFT):
            print("LEFT")
        if wapi.GetAsyncKeyState(RIGHT):
            print("RIGHT")
