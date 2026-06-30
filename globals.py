import cv2

SCALE_FACTOR  = 1
FACE_DETECTION_RATE = 5
FACE_RECOGNITION_RATE = 10
REGISTERED_FACES = []
RECOGNIZED_FACES = {}
FACE_TRACKERS= {}
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
