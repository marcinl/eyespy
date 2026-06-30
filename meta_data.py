import cv2
import dlib
import numpy as np 
import sys
import os

import emo
import gender
import face_pose
import uuid



class MetaData:
    def __init__(
                self, 
                identifier, 
                frame, 
                grey,
                mark_up, 
                face, 
                camera_matrix, 
                camera_distortion):
        self.identifier = identifier
        self.name = None
        self.x, self.y, self.w, self.h = face
        self.face_location = face
        self.roi = self.get_roi(face)
        self.emotion = None
        self.gender = None
        self.frame = frame
        self.grey = grey
        self.cropped_face = self.get_crop()
        self.mark_up = mark_up
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion

    def get_crop(self):
        x, y, w, h = self.x, self.y, self.w, self.h
        return self.grey[y:y+h,x:x+w]

    def get_roi(self, face_location):
        x, y, w, h = face_location
        roi = [x, y, x+w, y+h]
        return roi

    def update_position(self, face):
        self.roi = self.get_roi(face)
        self.x, self.y, self.w, self.h = face


    def get_emo(self):
        self.emotion = emo.detect_emo(self.grey, self.roi)
        print('**', self.emotion)

    def get_gender(self, model):
        self.gender = model.get_gender(self.face_location, self.grey)
        print('^^', self.gender)


    def draw_box(self):
        # draw box
        self.mark_up = face_pose.draw_3D(self)


    def get_pose(self):
        #self.imgpts = face_pose.detect_pose(self)
        self.dlib_landmarks = face_pose.detect_pose(self)
