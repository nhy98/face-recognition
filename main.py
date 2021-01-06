import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from libs.detector import _detector



cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    count = count+1
    if ret:
        # Our operations on the frame come here
        img_raw = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        img_raw = _detector(img_raw, count)
        # Display the resulting frame
        cv2.imshow('frame', img_raw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()