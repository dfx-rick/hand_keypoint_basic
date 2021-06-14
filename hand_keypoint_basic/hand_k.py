import cv2
import matplotlib.pyplot as plt
import numpy as np

pfile= 'hands/pose_deploy.prototxt'
wfile= 'hands/pose_iter_102000.caffemodel'
kpoints=22

img=cv2.imread('hand.jpg')

net= cv2.dnn.readNetFromCaffe(pfile, wfile)

imgWidth = img.shape[1]
imgHeight = img.shape[0]

aspect_ratio = imgWidth/imgHeight

iHeight = 368
iWidth = int(((aspect_ratio*iHeight)*8)//8)

iBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (iWidth, iHeight),(0, 0, 0), swapRB=False, crop=False)
net.setInput(iBlob)
output = net.forward()

points = []
frameCopy=cv2.imread('hand.jpg')
threshold= 0.2

for i in range(kpoints):
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (imgWidth, imgHeight))


    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    if prob > threshold :
        cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

 
        points.append((int(point[0]), int(point[1])))
    else :
        points.append(None)
        cv2.imshow('Output-Keypoints', frameCopy)
