from twine.detection import MobileNetDetector
from twine.tracking import CentroidTracker, Trackable
import twine.api as twine_api
import cv2
import dlib
import numpy as np

detector = MobileNetDetector()
trackers = []

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalDown = 0
totalUp = 0

def process(frame, rgb, track=False):
	global totalDown
	global totalUp
	global trackers

	(H, W) = frame.shape[:2]

	rects = []

	if track:
		trackers = []

		boxes = detector.detect(frame, W, H)

		for (startX, startY, endX, endY) in boxes:
			tracker = dlib.correlation_tracker()
			rect = dlib.rectangle(startX, startY, endX, endY)
			tracker.start_track(rgb, rect)

			trackers.append(tracker)
	else:
		for tracker in trackers:
			tracker.update(rgb)
			pos = tracker.get_position()

			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY))
	
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)

		if to is None:
			to = Trackable(objectID, centroid)
		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			if not to.counted:
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					twine_api.report_entrance()
					to.counted = True
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					twine_api.report_entrance()
					to.counted = True

		trackableObjects[objectID] = to

		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


	info = [
		("Up", totalUp),
		("Down", totalDown)
	]

	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

