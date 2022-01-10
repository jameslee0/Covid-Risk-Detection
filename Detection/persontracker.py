from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class PersonTracker():
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.social_dist = OrderedDict()
        self.bbox = OrderedDict()
        self.mask_box = OrderedDict()
        self.wearing_mask = OrderedDict()

    def register(self, centroid, soc_dist, box, mask_box, wearing_mask):
        # Registering every new user in the frame with their unique information stored
        self.objects[self.nextObjectID] = centroid
        self.social_dist[self.nextObjectID] = soc_dist
        self.disappeared[self.nextObjectID] = 0
        self.bbox[self.nextObjectID] = box
        self.mask_box[self.nextObjectID] = mask_box
        self.wearing_mask[self.nextObjectID] = wearing_mask
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        # Removing any users out of frame too long
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.social_dist[objectID]
        del self.bbox[objectID]
        del self.mask_box[objectID]
        del self.wearing_mask[objectID]

    def update(self, rects, soc_dist, mask_box, wearing_mask):
        if len(rects) == 0: #removing all users if noone detected in frame
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # Calculating centroids
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # Registering any new users if none are registered
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], soc_dist[i], rects[i], mask_box[i], wearing_mask[i])

        else:
            # Updating and matching users in frame based on centroid position
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                # Updating users information based on tracking
                objectID = objectIDs[row]
                self.mask_box[objectID] = mask_box[col]
                self.bbox[objectID] = rects[col]
                self.wearing_mask[objectID] = wearing_mask[col]
                self.social_dist[objectID] = soc_dist[col]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], soc_dist[col], rects[col], mask_box[col], wearing_mask[col])

        return self.objects

    # Getter functions
    def get_soc_dist(self, objectID):
        return self.social_dist[objectID]

    def get_bbox(self, objectID):
        return self.bbox[objectID]

    def get_mask_box(self, objectID):
        return self.mask_box[objectID]

    def get_wearing_mask(self, objectID):
        return self.wearing_mask[objectID]
