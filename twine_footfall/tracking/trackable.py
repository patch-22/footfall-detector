class Trackable:
    def __init__(self, objectId, centroid):
        self.objectId = objectId
        self.centroids = [centroid]

        self.counted = False