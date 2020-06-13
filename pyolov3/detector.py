from __future__ import division

import os
import pickle as pkl
import random
import shutil
from urllib import request

import cv2
import numpy as np
import torch
import wget
from torch.autograd import Variable

from .darknet import Darknet
from .util import load_classes, write_results

model_path = "{}/.config/pyyolov3".format(os.path.expanduser("~"))
file_url = {
    "yolov3.weights": "https://dl.dropboxusercontent.com/s/gx6bgx66dgnzblx/yolov3.weights",
    "yolov3-openimages.weights": "https://dl.dropboxusercontent.com/s/4zkw9tnlzdeae3d/yolov3-openimages.weights",
    "yolov3-face.weights": "https://dl.dropboxusercontent.com/s/7el6s0tju30oldw/yolov3-face.weights",
}
file_dir = os.path.dirname(__file__)


class Detector:
    """YOLO Detector"""

    def __init__(
        self, cfgfile, weightsfile, classfile, confidence, reso=160, nms_thesh=0.4
    ):
        self.cuda = torch.cuda.is_available()

        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)
        self.classes = load_classes(classfile)
        self.num_classes = len(self.classes)
        self.bbox_attrs = 5 + self.num_classes

        self.confidence = confidence
        self.nms_thesh = nms_thesh

        self.model.net_info["height"] = reso
        self.inp_dim = int(self.model.net_info["height"])

        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        if self.cuda:
            self.model.cuda()

        self.model.eval()

        self.colors = pkl.load(open("{}/data/pallete".format(file_dir), "rb"))

    def detect(self, frame):
        """detect objects"""
        img, orig_im, dim = self._prep_image(frame, self.inp_dim)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if self.cuda:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = self.model(Variable(img), self.cuda)
        output = write_results(
            output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh
        )

        if type(output) == int:
            return orig_im, None
        else:
            output[:, 1:5] = (
                torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim
            )

            im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            result = []
            detimg = np.copy(orig_im)
            for d in output:
                detimg, r = self._write(d, detimg)
                if r is not None:
                    result.append(r)

            return detimg, result

    def _prep_image(self, img, inp_dim):
        """Prepare image for inputting to the neural network."""
        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = cv2.resize(orig_im, (self.inp_dim, self.inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    def _write(self, x, img):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        if c1 == c2:
            return img, None
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])
        color = random.choice(self.colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c3 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c3, color, -1)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            [225, 255, 255],
            1,
        )
        confidence = float(x[5] + x[6]) / 2
        r = {
            "p1": (int(c1[0]), int(c1[1])),
            "p2": (int(c2[0]), int(c2[1])),
            "class": label,
            "confidence": confidence,
        }
        return img, r


def get_detector(model_name, confidence, reso=160, nms_thesh=0.4):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if model_name == "coco":
        cfg_file = "yolov3.cfg"
        class_file = "coco.names"
        weights_file = "yolov3.weights"
    elif model_name == "openimages":
        cfg_file = "yolov3-openimages.cfg"
        class_file = "openimages.names"
        weights_file = "yolov3-openimages.weights"
    elif model_name == "widerface":
        cfg_file = "yolov3-face.cfg"
        class_file = "face.names"
        weights_file = "yolov3-face.weights"
    else:
        print("Undefinded : {}".format(model_name))
        return None

    if not os.path.exists(model_path + "/" + weights_file):
        print("Get weights_file : {}".format(weights_file))
        wget.download(file_url[weights_file], model_path)

    return Detector(
        file_dir + "/data/" + cfg_file,
        model_path + "/" + weights_file,
        file_dir + "/data/" + class_file,
        confidence,
        reso=reso,
        nms_thesh=nms_thesh,
    )
