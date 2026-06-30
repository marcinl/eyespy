# import face_recognition
import cv2
import face_recognition
from collections import namedtuple

REGISTERED_FACES = []
RECOGNIZED_FACES = {}
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




def registerFace(img, name):
    face = namedtuple("face", "id name face_encodings")
    target_image = face_recognition.load_image_file(img)
    target_face_encoding = face_recognition.face_encodings(target_image)[0]
    face.face_encodings = target_face_encoding 
    face.name = name
    REGISTERED_FACES.append(face)

registerFace("./known_faces/Marcin.jpg", "Marcin")
registerFace("./known_faces/Irah.jpg", "Irah")
registerFace("./known_faces/Shane.jpg", "Shane")

# Recognize faces
def recognizeFace(face):
    recognizedFace = None
    #if faceNames[fid] == None:
    #    faceNames[fid] = "Unknown"
    face_locations = face_recognition.face_locations(face, 1, model="hog")
    print("Face locations {}".format(face_locations))
    face_encodings = face_recognition.face_encodings(face, face_locations)

    for regFace in REGISTERED_FACES:
        #print("Checking face '{}'".format(regFace.name))
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([regFace.face_encodings], face_encoding)
            if match[0]:
                print("Found face '{}'".format(regFace.name))
                recognizedFace = regFace.name
    return recognizedFace

def recognizeFaces(frame, face_locations):
    recognised_faces = []

    fid = 1
    for (x, y, w, h) in face_locations:
        face = frame[y: y + h, x: x + w]
        name = recognizeFace(face)
        if name != None:
            recognised_faces.append((name, (x, y, w, h)))
        else:
            unknown_name = "Face #{}".format(fid)
            recognised_faces.append((unknown_name, (x, y, w, h)))
        fid += 1

    return recognised_faces


def get_face_locations(img):
    # output [[face_location],[x,y,w,h]] or []
    face_locations = FACE_DETECTOR.detectMultiScale(img, 1.3, 5, minSize=(48, 48))
    if len(face_locations):
        return face_locations
    else:
        return []

def get_identifiers(face_locations, face_IDs, num_frames, frame):
    recognized_faces = []
    if len(face_locations):
        
        if num_frames % 5 == 0:
            recognized_faces = recognizeFaces(frame, face_locations)
        else:
            
    return recognized_faces

def get_identifiers_old(face_locations, num_frames, frame, DATA):
    recognized_faces = []
    if len(face_locations):
        
        if num_frames % 5 == 0:
            recognized_faces = recognizeFaces(frame, face_locations)
        else:
            minimums = []
            for identifier, meta in DATA.items():
                old_x = meta.x
                x_list = [face[0] for face in face_locations]
                distance = [abs(new_x - old_x) for new_x in x_list]
                min_distance = min(distance)
                min_index = distance.index(min_distance)
                minimums.append((min_index, min_distance, identifier))
            distance = {}
            tracker = {}

            for (min_index, min_distance, identifier) in minimums:
                if identifier not in tracker:
                    distance[identifier] = min_distance
                    tracker[identifier] = min_index
                else:
                    if distance[identifier] > min_distance:
                        distance[identifier] = min_distance
                        tracker[identifier] = min_index

            s = set()
            for identifier, min_index in tracker.items():
                face = tuple(face_locations[min_index])
                if face not in s:
                    s.add(face)
                    recognized_faces.append((identifier, face))
                else:
                    DATA.pop(identifier)

    return recognized_faces, DATA