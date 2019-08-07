import cv2
import numpy as np
import dlib

CONFIDENCE_THRESHOLD = 0

class MobileNetDetector:
    def __init__(self):
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        self.network = cv2.dnn.readNetFromCaffe('twine/model/MobileNetSSD_deploy.prototxt', 'twine/model/MobileNetSSD_deploy.caffemodel')
        self.target_class = 'person'

    def detect(self, frame, W, H):
        blob = cv2.dnn.blobFromImage(frame, 0.0078843, (W, H), 127.5)
        self.network.setInput(blob)

        detections = self.network.forward()

        boxes = []

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                idx = int(detections[0, 0, i, 1])

                if self.classes[idx] != self.target_class:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                boxes.append(box.astype('int'))

        return boxes
