import cv2
from glob import glob
import os
import random

path_P = './task7_insulator/P'
path_N = './task7_insulator/N'

def readImg_P():
    P = glob(os.path.join(path_P, '*.jpg'))
    N = glob(os.path.join(path_N, '*.jpg'))
    while 1:
        random.shuffle(P)
        for imgPath in P:
            img = cv2.imread(os.path.join(imgPath))
            img = cv2.resize(img, (448, 448))/255.0
            yield img


def readImg_N():
    N = glob(os.path.join(path_N, '*.jpg'))
    while 1:
        random.shuffle(N)
        for imgPath in N:
            img = cv2.imread(os.path.join(imgPath))
            img = cv2.resize(img, (448, 448))/255.0
            yield img
