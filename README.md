# A PyTorch implementation of a YOLO v3 Object Detector

[YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)のPyTorch実装版です。  
[ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)の実装を活用させていただいています。

## 導入方法

```bash
pip install pyolov3
```

## 使い方

- Webカメラを使ったサンプルコード

```python
import cv2

from pyolov3 import get_detector

yolo = get_detector("coco", 0.5) # 使用したい学習済みモデルとConfidenceの閾値を設定
cap = cv2.VideoCapture(0)

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
```

## 使用できる学習済みモデル

現状は以下のモデルを指定できます。

- [MS COCO](http://cocodataset.org/)
  - 80クラス検出モデル
  - `Detector("coco", confidence)`と指定
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
  - 600クラス検出モデル
  - `Detector("openimages", confidence)`と指定
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
  - 顔検出モデル
  - `Detector("widerface", confidence)`と指定
