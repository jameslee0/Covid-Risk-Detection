from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os

def preprocess_image(image):
  image = np.array(image)
  img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
  image = tf.image.convert_image_dtype(img_reshaped, tf.float32, name="images")  
  return image

def mask_detect(mask_points, net, frame):
    wearing_mask = "unknown"
    
    (x, y, w, h) = mask_points #Retrieve mask points
    roi_face = frame[int(y - 2 * w / 5):int(y + 3 * w / 5), x:(x + w)] #Cropping face

    HEIGHT, WIDTH = roi_face.shape[:2]
    if HEIGHT >= 32 and WIDTH >= 32: #Only detecting face images 32X32 or larger
        save_loc = os.path.join("./mask_images", '_head_shot.jpg')
        cv2.imwrite(save_loc, roi_face)

        pic = cv2.imread(save_loc)
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic,(128,128))

        pic = preprocess_image(pic)
        prediction = np.argmax(net.predict(pic))

        if prediction == 0:
            wearing_mask = "nomask"
        elif prediction == 1:
            wearing_mask = "mask"
        else:
            wearing_mask = "unknown"
            

    return wearing_mask