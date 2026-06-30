import cv2
import dlib
import numpy as np 
import sys
import os

import myVGG as vgg

MODEL = vgg.VGG_16('my_model_weights_83.h5')


def preprocess_face(img, roi, face_shape=(48, 48)):
    face_scaled = None
    okay = True
    crop = img[roi[1]:roi[3], roi[0]:roi[2]]
    try:
        face_scaled = cv2.resize(crop, face_shape)
    except:
        print('hit exception in preprocessing face')
        okay = False
    return okay, face_scaled

def detect_emo(grey, roi):
    emo = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    okay, face = preprocess_face(grey, roi)
    if okay:
        test = np.expand_dims(face, axis=0)
        test = np.expand_dims(test, axis=0)
        result = MODEL.predict(test)[0]
        index = np.argmax(result)
        # print(emo[index], 'prob:', max(result))
        return emo[index]
    else:
        return None