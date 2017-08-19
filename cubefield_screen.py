import numpy as np
import cv2
import time
from cubefield_inputkeys import PressKey, ReleaseKey, LEFT, RIGHT
from cubefield_grabscreen import grab_screen

b_left = 1320
b_top = 40
b_right = 1920
b_bottom = 440

a_right = b_right - b_left
a_bottom = b_bottom - b_top

roi_vertices = np.array([[0, a_bottom], [0, int(a_bottom*0.5)], [a_right, int(a_bottom*0.5)], [a_right, a_bottom]])

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(o_image):
        p_image = cv2.Canny(o_image, threshold1=200, threshold2=300)        
        p_image = roi(p_image, roi_vertices)
        return p_image


def get_img():
    screen = grab_screen(region=(b_left, b_top, b_right, b_bottom))
    new_screen = process_img(screen)

    #cv2.imshow('window', new_screen)

    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #    cv2.destroyAllWindows()

    return new_screen, screen

if __name__ == '__main__':
    while(True):
        screen = grab_screen(region=(b_left, b_top, b_right, b_bottom))
        new_screen = process_img(screen)

        cv2.imshow('window', screen)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
