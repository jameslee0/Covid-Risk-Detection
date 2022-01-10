import numpy as np
import cv2

def detect_persons(frame, net, ln, pid=0):
    HEIGHT_F, WIDTH_F, CHANNEL_F = frame.shape
    MIN_CONFIDENCE, NMS_THRESHOLD = 0.7, 0.3

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(ln)

    detections = []
    boxes = []
    boxes_mask = []
    centroids = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == pid and confidence > MIN_CONFIDENCE:
                center_x = int(detection[0] * WIDTH_F)
                center_y = int(detection[1] * HEIGHT_F)

                w = int(detection[2] * WIDTH_F)
                h = int(detection[3] * HEIGHT_F)
                w_mask = int(detection[2] * WIDTH_F * .75)
                h_mask = int(detection[3] * HEIGHT_F * .75)

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                x_mask = int(center_x - (w_mask / 2))
                y_mask = int(center_y - (h_mask / 2))

                boxes.append([x, y, w, h])
                boxes_mask.append([x_mask, y_mask, w_mask, h_mask])
                centroids.append((center_x, center_y))
                confidences.append(float(confidence))
                
                

    index = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    if len(index) > 0:
        for i in index.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #print(boxes_mask[i][:4], "start")
            (x_mask, y_mask) = (boxes_mask[i][0], boxes_mask[i][1])
            (w_mask, h_mask) = (boxes_mask[i][2], boxes_mask[i][3])

            det = ((x_mask, y_mask, w_mask, h_mask), (x, y, w + x, h + y), centroids[i])
            detections.append(det)

    return detections