import cv2
import dlib
import numpy as np 
import sys
import os
import globals

# TODO: find all occurences of Haar detector and point to this
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#RECOGNIZED_FACES = {}
#REGISTERED_FACES = []


def get_face_locations(frame, frame_no):
    
    currentFaceID = 0

    # rectangles surrounding faces
    facesLocations = []
    faceIds = []

    # Resize the image to 180x320
    rows, cols, chans = frame.shape
    baseImage = cv2.resize(frame, (int(cols / globals.SCALE_FACTOR), int(rows / globals.SCALE_FACTOR)), \
                            interpolation = cv2.INTER_CUBIC)

    # Increase the framecounter
    frame_no += 1 



    #Update all the trackers and remove the ones for which the update
    #indicated the quality was not good enough
    fidsToDelete = []
    for fid in globals.FACE_TRACKERS.keys():
        trackingQuality = globals.FACE_TRACKERS[fid].update(baseImage)

        # If the tracking quality is good enough, we must delete
        # this tracker
        if trackingQuality < 5:
            fidsToDelete.append(fid)

    for fid in fidsToDelete:
        print("Removing fid " + str(fid) + " from list of trackers")
        globals.FACE_TRACKERS.pop(fid , None)
        globals.RECOGNIZED_FACES.pop(fid, None)



    # At specified rate we will have to determine which faces
    # are present in the frame
    if (frame_no % globals.FACE_DETECTION_RATE) == 0:

        # Use gray-based image for face detection
        #gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        # equalise image
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (48, 48))
        #eqImage = clahe.apply(gray)
        # Now use the haar cascade detector to find all face
        # in the image
        faces = globals.FACE_DETECTOR.detectMultiScale(baseImage, 1.3, 5)



        # Loop over all faces and check if the area for this
        # face is the largest so far
        for (_x,_y,_w,_h) in faces:
            x = int(_x)
            y = int(_y)
            w = int(_w)
            h = int(_h)


            # calculate the centerpoint
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h



            # Variable holding information which faceid we 
            # matched with
            matchedFid = None

            # Now loop over all the trackers and check if the 
            # centerpoint of the face is within the box of a 
            # tracker
            for fid in globals.FACE_TRACKERS.keys():
                tracked_position =  globals.FACE_TRACKERS[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())


                # calculate the centerpoint
                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                # check if the centerpoint of the face is within the 
                # rectangleof a tracker region. Also, the centerpoint
                # of the tracker region must be within the region 
                # detected as a face. If both of these conditions hold
                # we have a match
                if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                     ( t_y <= y_bar   <= (t_y + t_h)) and 
                     ( x   <= t_x_bar <= (x   + w  )) and 
                     ( y   <= t_y_bar <= (y   + h  ))):
                     matchedFid = fid
                     # Face recognition
                     #currentFace = frame[scale_factor * y - 20: scale_factor * (y + h) + 20, \
                     #                   x * scale_factor  - 10 : (x + w) * scale_factor  + 20]
                     #currentFace = baseImage[ y - 20: y + h + 20, x - 10 : x + w  + 20]
                     #t = threading.Thread(target = recognizeFace ,
                     #                     args=(currentFace, RECOGNIZED_FACES, fid))
                     #t.start()

                # If no matched fid, then we have to create a new tracker
            if matchedFid is None:

                print("Creating new tracker " + str(currentFaceID))

                # Create and store the tracker 
                tracker = dlib.correlation_tracker()
                #tracker.start_track(baseImage,
                #                        dlib.rectangle(x-10,
                #                                    y-20,
                #                                    x+w+10,
                #                                   y+h+20))
                tracker.start_track(baseImage,
                                       dlib.rectangle(x, y, x + w + 10, y + h + 20))

                globals.FACE_TRACKERS[ currentFaceID ] = tracker

                #Increase the currentFaceID counter
                currentFaceID += 1




    # Now loop over all the trackers to get the rectangles
    # around the detected faces. 
    for fid in globals.FACE_TRACKERS.keys():
        tracked_position =  globals.FACE_TRACKERS[fid].get_position()

        t_x = int(tracked_position.left() * globals.SCALE_FACTOR)
        t_y = int(tracked_position.top() * globals.SCALE_FACTOR)
        t_w = int(tracked_position.width() * globals.SCALE_FACTOR)
        t_h = int(tracked_position.height() * globals.SCALE_FACTOR)

        facesLocations.append((t_x, t_y, t_w, t_h))
        faceIds.append(fid)
        
    return facesLocations, faceIds