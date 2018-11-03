# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import cv2
import gc
import numpy as np
import time
from train_MNIST import Model, MODEL_PATH

IMAGE_SIZE = 28

def resize_image(frame, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    #get the size of image
    h, w = frame.shape

    #if width != height
    #get the longer one
    longest_edge = max(h, w)

    #calculate how much pixel should be add to the shorter side
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    #RGB
    #set the border color
    BLACK = [0, 0, 0]

    #border
    constant = cv2.copyMakeBorder(frame, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)

    #resize image & return
    return cv2.resize(constant, (height, width))

if __name__ == '__main__':
    model = Model()
    model.load_model(MODEL_PATH)
    if sys.argv[1] == '--image_path':
        #set the color of bounding boxes
        color = (255,0,255)

        image = cv2.imread(sys.argv[2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = resize_image(image)
        print 'the size of input image is: '
        print image.shape
        numID = model.predict(image)
        print '[]~(￣▽￣)~*检测中~'
        print numID

    if sys.argv[1] == '--camera_id':
        #using camera to do real-time detect
        cap = cv2.VideoCapture(int(sys.argv[2]))
        time.sleep(3)
        delay = 11
        curr_frame = 1
        while True:
            _,frame = cap.read()
            if curr_frame % delay == 0:
                cv2.imshow("number_detect", frame)
                #graying, reshape channel=3 to channel=1
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = resize_image(frame_gray)

                print 'the size of input image is:'
                print image.shape
                numID = model.predict(image)
                print '[]~(￣▽￣)~*检测中~'
                print numID
            curr_frame += 1

            #等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            #如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break

        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()
