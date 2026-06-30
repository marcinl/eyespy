import cv2
import os
import numpy as np


IMAGE_PATH = './known_faces/'


class Recognizer:
    def __init__(self):
        self.model = cv2.face.createLBPHFaceRecognizer(neighbors=8, threshold=10.0)
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
        self.name_lookup = {}
        self.init_train()


    def get_path_list(self):
        f = []
        for (dirpath, dirnames, filenames) in os.walk(IMAGE_PATH):
            f.extend(filenames)
        return f

    def init_train(self):
        files = self.get_path_list()
        print(files)
        cropped_faces = []
        labels = []
        for i, file_name in enumerate(files):
            image_path = IMAGE_PATH + file_name
            name = file_name.split('.')[0]
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_numpy = np.array(img,'uint8')
            faces = self.detector.detectMultiScale(img_numpy)
            if len(faces):
                (x,y,w,h) = faces[0]
                crop = img_numpy[y:y+h,x:x+w]
                cropped_faces.append(crop)
                labels.append(i)
                self.name_lookup[i] = name
        self.model.train(cropped_faces, np.array(labels))


    def get_face_locations(self, img):
        # output [[face_location],[x,y,w,h]] or []
        face_locations = self.detector.detectMultiScale(img, 1.3, 5, minSize=(48, 48))
        if len(face_locations):
            return face_locations
        else:
            return []

    def get_identities(self, face_locations, grey):
        l = []
        for (x,y,w,h) in face_locations:
            crop = grey[y:y+h,x:x+w]
            identity = self.model.predict(crop)
            l.append((self.name_lookup[identity], (x,y,w,h)))
        return l



