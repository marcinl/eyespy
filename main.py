import cv2
import dlib
import numpy as np 
import sys
import os

import time
import uuid
from functools import partial

import face_tracker
import face_pose
import emo
import gender
from meta_data import MetaData
from collections import namedtuple
import recognizer

REGISTERED_FACES = []
GENDER_MODEL = gender.GenderClassifier()

RUNNING = True
DATA = {}


def fill_meta(recognized_faces, 
                data, 
                frame, 
                grey, 
                mark_up, 
                camera_matrix, 
                camera_distortion
                ):
    
    identifier = recognized_faces[0]
    name = recognized_faces[1]
    face = recognized_faces[2]
    #print("{} and data {}".format(identifier, data))
    if not identifier in data:
        meta = MetaData(identifier=identifier, 
                        frame=frame, 
                        grey=grey, 
                        mark_up=mark_up,
                        face=face, 
                        camera_matrix=camera_matrix, 
                        camera_distortion=camera_distortion)
        DATA[identifier] = meta
        meta.get_gender(GENDER_MODEL)
    else:
        meta = data[identifier]
        meta.update_position(face)
    meta.frame = frame
    meta.grey = grey
    meta.name = name
    meta.get_emo()
    meta.get_pose()
    meta.draw_box()
    return (identifier, meta)



def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(48,48))
    clahe_image = clahe.apply(gray)
    return clahe_image



if __name__ == '__main__':
    #global RUNNING
    

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    cam_w = int(cap.get(3))
    cam_h = int(cap.get(4))
    camera_matrix, camera_distortion = face_pose._initCamMatrix(cam_w=cam_w, cam_h=cam_h)
    num_frames = 0

    # register known faces

    
    frame_times = []   
    start = time.time()
    first_frame_start = start
    while(RUNNING):
        # calculate fps

        ret, frame = cap.read()
        mark_up = frame.copy()

        if not ret:
            continue
        
        gray = preprocess(frame)

        # img, faces_locations, face_ids = face_tracker.get_face_locations(frame, num_frames) # TODO return face dict
        face_locations = recognizer.get_face_locations(frame)
        #img, faces_locations, face_ids = face_tracker.get_face_locations(frame, num_frames) # TODO return face dict
        # faces_locations = get_face_locations_old(frame)

        recognized_faces, DATA = recognizer.get_identifiers_old(face_locations, num_frames, frame, DATA)
        #recognized_faces = recognizer.get_identifiers(face_locations, num_frames, frame)

        recognised_faces_with_ids = []
        fid = 0
        if len(recognized_faces) > 0:
            for name, (x, y, w, h) in recognized_faces:
                recognised_faces_with_ids.append(("Face {}".format(fid), name, (x, y, w, h)))
                fid += 1





        metas = list(map(partial(fill_meta, 
                            data=DATA, 
                            frame=frame, 
                            grey=gray, 
                            mark_up=mark_up,
                            camera_matrix=camera_matrix, 
                            camera_distortion=camera_distortion), 
                    recognised_faces_with_ids))


       

        # timing the main loop
        end = time.time() 
        delta = time.time() - start
        start = end
        frame_times.append(delta)
        frame_time = frame_times[-20:0]
        fps = len(frame_times) / sum(frame_times)
        print("FPS {}".format(fps))

        # blend face information with the frame
        alpha = 0.4
        output = cv2.addWeighted(mark_up, alpha, frame, 1 - alpha, 0)
        cv2.imshow('Video', output)
        num_frames += 1
        # wait for 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            RUNNING = False


    seconds = time.time() - first_frame_start        
    fps_avg  = num_frames / seconds;
    print("Estimated frames per second : {0}".format(fps_avg))

    cap.release()
    cv2.destroyAllWindows() 


