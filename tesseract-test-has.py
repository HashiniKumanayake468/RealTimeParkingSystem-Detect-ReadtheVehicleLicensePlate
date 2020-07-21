import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pytesseract

im = cv2.imread('testout.png')
# grayscale to binary
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, im_bw = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# invert image
im_bw = 255-im_bw
# find contours
cnts, hierarchy = cv2.findContours(im_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# remove small components based on are
if cnts is not None:
    for i in range(0, len(cnts)):
        a = cv2.contourArea(cnts[i])
        if a < 7:
            cv2.drawContours(im_bw, cnts, i, 0, cv2.FILLED)
im_bw = 255-im_bw
print(pytesseract.image_to_string(im_bw))
    #use tesseract to convert image into string
    # text = pytesseract.image_to_string(bw_img,lang='eng')
    # print("Number is:",text)
