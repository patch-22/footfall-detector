import numpy as np
import imutils
import cv2
import time

from imutils.video import VideoStream
from itertools import count

from twine.detection import MobileNetDetector
from twine.processing import process

# Take video from the webcam
vs = VideoStream(src=0).start()
time.sleep(2.0)

network = MobileNetDetector()

for current_frame in count():
	frame = vs.read()

	# Resize frame for performance
	frame = imutils.resize(frame, width=300)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	(H, W) = frame.shape[:2]
	
	# Retrack every 5 frames
	if current_frame % 5 == 0:
		process(frame, rgb, True)
	else:
		process(frame, rgb, False)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

vs.stop()
vs.release()
cv2.destroyAllWindows()
