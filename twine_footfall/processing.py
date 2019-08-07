from twine_footfall.tracking import Trackable, CentroidTracker
from twine_footfall.detection import MobileNetDetector
import twine_footfall.api as twine_footfall_api

import cv2
import dlib
import numpy as np

class FootfallProcessor:
	def __init__(self, frame):
		self.totalIn = 0
		self.totalOut = 0

		self.trackers = []
		self.trackedObjects = {}

		self.centroid_tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)
		self.person_detector = MobileNetDetector()

		(self.H, self.W) = frame.shape[:2]



	def process_frame(self, frame, track=False):
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		rects = []

		if track:
			self.trackers = []

			boxes = self.person_detector.detect(frame, self.W, self.H)

			for (startX, startY, endX, endY) in boxes:
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				self.trackers.append(tracker)			
		else:
			for tracker in self.trackers:
				tracker.update(rgb)
				pos = tracker.get_position()

				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				rects.append((startX, startY, endX, endY))
		
		cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (255, 255, 255), 2)
		objects = self.centroid_tracker.update(rects)

		for (objectID, centroid) in objects.items():
			to = self.trackedObjects.get(objectID, None)

			if to is None:
				to = Trackable(objectID, centroid)
			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				if not to.counted:
					if direction < 0 and centroid[1] < self.H // 2:
						self.handle_in()
						to.counted = True
					elif direction > 0 and centroid[1] > self.H // 2:
						self.handle_out()
						to.counted = True

			self.trackedObjects[objectID] = to

			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		self.display_info(frame)

	def display_info(self, frame):
		info = [
			("In", self.totalIn),
			("Out", self.totalOut)
		]

		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

	def handle_in(self):
		self.totalIn += 1
		twine_footfall_api.report_entrance()	
	
	def handle_out(self):
		self.totalOut += 1
		twine_footfall_api.report_exit()