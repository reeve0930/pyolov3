from __future__ import division

import cv2
import numpy as np

from detector import Detector

if __name__ == "__main__":
    cfgfile = "cfg/yolov3-face.cfg"
    weightsfile = "weights/yolov3-face.weights"
    namefile = "data/face.names"

    yolo = Detector(cfgfile, weightsfile, namefile, 0.9)

    cap = cv2.VideoCapture(2)

    assert cap.isOpened(), "Cannot capture source"

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img, result = yolo.detect(frame)
            cv2.imshow("frame", img)
            print(result)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
        else:
            break
