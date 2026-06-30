import cv2
import os
import globals
import numpy as np


MALE_PATH = './gender/male/'
FEMALE_PATH = './gender/female/'

FEMALE_CHECK = './gender_check/female/'
MALE_CHECK = './gender_check/male/'

class GenderClassifier:
    def __init__(self):
        #self.model = cv2.face.createFisherFaceRecognizer()
        self.model = cv2.face.FisherFaceRecognizer_create()
        self.detector = globals.FACE_DETECTOR
        self.init_train()

    def init_train(self):
        faces, labels = self.get_faces_labels(FEMALE_PATH, MALE_PATH) 
        self.model.train(faces, np.array(labels))

    def get_path_list(self, female, male):
        f = []
        m = []
        for (dirpath, dirnames, filenames) in os.walk(female):
            f.extend(filenames)
        for (dirpath, dirnames, filenames) in os.walk(male):
            m.extend(filenames)
        return f, m

    

    def get_faces_labels(self, f_path, m_path, crop=False):

        def cropper(img_numpy):
            faces = self.detector.detectMultiScale(img_numpy)
            if len(faces):
                (x,y,w,h) = faces[0]
                crop = img_numpy[y:y+h,x:x+w]
                crop = cv2.resize(crop, (64,64))
                return True, crop
            else:
                return False, None

        def img_lab(path, f, faces, labels, label_type, crop):
            for i, file_name in enumerate(f):
                image_path = path + file_name
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                img_numpy = np.array(img,'uint8')
                if crop:
                    ret, cropped = cropper(img_numpy)
                    if ret:
                        img_numpy = cropped
                faces.append(img_numpy)
                labels.append(label_type)
            return faces, labels

        f, m = self.get_path_list(f_path, m_path)
        labels = []
        faces = []
        faces, labels = img_lab(f_path, f, faces, labels, 1, crop)
        faces, labels = img_lab(m_path, m, faces, labels, 0, crop)
        return faces, labels

        


    def get_face_locations(self, img):
        # output [[face_location],[x,y,w,h]] or []
        face_locations = self.detector.detectMultiScale(img, 1.3, 5, minSize=(48, 48))
        if len(face_locations):
            return face_locations
        else:
            return []

    def get_gender(self, face_location, grey):
        gender = {1: 'Male', 0: 'Female', -1: 'Unkown'}
        l = []
        x,y,w,h = face_location
        crop = grey[y:y+h,x:x+w]
        crop = cv2.resize(crop, (64, 64))
        gender_type, confidence = self.model.predict(crop) # 0 = male, 1 = female
        return gender[gender_type]

    def accuracy(self):
        f, m = self.get_path_list(FEMALE_CHECK, MALE_CHECK)
        file_names = f + m
        faces, ground_truth = self.get_faces_labels(FEMALE_CHECK, MALE_CHECK, crop=True)
        predictions = []
        accuracy = 0
        for i in range(len(faces)):
            gender_type = self.model.predict(faces[i])
            predictions.append(gender_type)
            if gender_type == ground_truth[i]:
                accuracy += 1
            else:
                print(file_names[i]) 
        accuracy /= len(faces)
        return accuracy


if __name__ == '__main__':
    model = GenderClassifier()
    accuracy = model.accuracy()
    print(accuracy)





