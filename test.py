import cv2

from pyolov3 import get_detector

yolo = get_detector("coco", 0.3)
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()

    detimg, result = yolo.detect(frame)
    print(result)

    cv2.imshow("test", detimg)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
