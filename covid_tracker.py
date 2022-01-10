import os
#Disable tensorflow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from config import DISTANCE, RESIZE_WIDTH, VIDEO_OUTPUT, FRAME_RATE
from Detection.persontracker import PersonTracker
from Detection.mask_detection import mask_detect
from Detection.detection import detect_persons
from scipy.spatial import distance as dist
from tensorflow import keras
import numpy as np
import argparse
import imutils
import time
import cv2

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="Path to input video file")
ap.add_argument("-s", "--save", type=bool, default=False, 
    help="(Optional) Video save")
ap.add_argument("-sd", "--socialdistance", type=bool, default=False,
    help="(Optional) Only run social distance portion")
ap.add_argument("-fm", "--facemask", type=bool, default=False,
    help="(Optional) Only run face mask detection portion")
args = vars(ap.parse_args())

# Loading in YOLO, coco, and facemask detector
print("[INFO] Loading YOLO Weights...")
net = cv2.dnn.readNetFromDarknet("models/yolov3.cfg", "models/yolov3.weights") #YOLO

print("[INFO] Loading COCO Names...")
LABELS = open("models/coco.names").read().strip().split("\n")#COCO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#mask_detection = keras.models.load_model("models/mask_detection") #facemask detector
print("[INFO] Loading Mask Detection Model...")
mask_detection = keras.models.load_model("models/mask_detection")

#Unique person tracker
personDetection = PersonTracker() #from Unique ID package

video_path = VIDEO_OUTPUT
cap = cv2.VideoCapture(args["video"])

if args["save"]:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(VIDEO_OUTPUT,cv2.VideoWriter_fourcc('M','J','P','G'), FRAME_RATE, (frame_width,frame_height))
    print("[INFO] Because the video is being saved, the window will not resize")

print("[INFO] Starting Detection...")
print("[INFO] Press 'q' at any moment to quit video")
while True:
    _, frame = cap.read()

    if not args["save"]:
        frame = imutils.resize(frame, width=RESIZE_WIDTH) #resize to fit screen window
    
    detections = detect_persons(frame, net, ln, pid=LABELS.index("person"))

    violate = set() #Holds social distance violations

    if len(detections) >= 2:
        centroids = np.array([det[2] for det in detections])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # Gets distance between two centroids and determined if they are social distancing
        for i in range(0, D.shape[1]):
            for j in range (i + 1, D.shape[1]):
                if D[i,j] < DISTANCE:
                    violate.add(i)
                    violate.add(j)
    
    mask_box = []
    box = []
    soc_dist = []
    wearing_mask = [] 
    for (i, (m_bbox, bbox, centroid)) in enumerate(detections):
        (x, y, w, h) = bbox
        box.append(bbox)

        if i in violate:
            soc_dist.append(False)
        else:
            soc_dist.append(True)

        (x_mask, y_mask, w_mask, h_mask) = m_bbox #Box for person
        mask_box.append([x_mask, y_mask, w_mask, h_mask]) #Box for persons head region

        wearing_mask.append(mask_detect(m_bbox, mask_detection, frame))

    people = personDetection.update(box, soc_dist, mask_box, wearing_mask)
    for (objectID, centroid) in people.items():
        (x, y, w, h) = personDetection.get_bbox(objectID)
        sd = personDetection.get_soc_dist(objectID)
        (x_mask, y_mask, w_mask, h_mask) = personDetection.get_mask_box(objectID)
        wm = personDetection.get_wearing_mask(objectID)
        
        if sd:
            soc_dist_col = (0, 255, 0) #Green
        else:
            soc_dist_col = (0, 0, 255) #Red

        if wm == "mask":
            mask_col = (0, 255, 0) #Green
        elif wm == "nomask":
            mask_col = (0, 0, 255) #Red
        else:
            mask_col = (255, 255, 255) #White

        if not args["facemask"]: #Face mask bounding box
            cv2.rectangle(frame, (x_mask, int(y_mask - 2 * w_mask / 5)), (x_mask + w_mask, int(y_mask + 3 * w_mask / 5)), mask_col, 2)

        if not args["socialdistance"]: #Social distance bounding box
            cv2.rectangle(frame, (x, y), (w, h), soc_dist_col, 2)

        # Centroid
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    
    cv2.imshow("Frame", frame)

    if args["save"]:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if args["save"]:
    out.release()
    print("[INFO] The video has been saved")

cap.release()
cv2.destroyAllWindows()
