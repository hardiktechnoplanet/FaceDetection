#!/usr/bin/env python
import cv2
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load a model imported from Tensorflowload using OpenCV dnn module
cvNet=cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Input image
img = cv2.imread(args["image"])
rows = img.shape[0]
cols = img.shape[1]

# Use the given image as input, which needs to be blob(s).
cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

# Runs a forward pass to compute the net output
cvOut = cvNet.forward()

# Loop on the outputs
for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
