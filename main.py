import numpy as np
import imutils
import cv2
import time

from imutils.video import VideoStream
from itertools import count

from twine_footfall.detection import MobileNetDetector
from twine_footfall.processing import FootfallProcessor

# Take video from the webcam
vs = VideoStream(src=0).start()
time.sleep(2.0)

network = MobileNetDetector()
processor = None

for current_frame in count():
	frame = vs.read()

	# Resize frame for performance
	frame = imutils.resize(frame, width=300)

	if processor is None:
		processor = FootfallProcessor(frame)
	
	# Retrack every 5 frames
	track = current_frame % 5 == 0
	processor.process_frame(frame, track)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

vs.stop()
vs.release()
cv2.destroyAllWindows()
