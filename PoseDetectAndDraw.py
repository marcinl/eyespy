import numpy as np
import cv2
#import sys
#import os
import dlib
#import glob
#from skimage import io
#import imutils
#from imutils import face_utils
import time



def _initCamMatrix( cam_w, cam_h):
    #Defining the camera matrix.
    #To have better result it is necessary to find the focal
    # lenght of the camera. fx/fy are the focal lengths (in pixels) 
    # and cx/cy are the optical centres. These values can be obtained 
    # roughly by approximation, for example in a 640x480 camera:
    # cx = 640/2 = 320
    # cy = 480/2 = 240
    # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
    c_x = cam_w / 2
    c_y = cam_h / 2
    angle = 50
    f_x = c_x / np.tan(angle/2 * np.pi / 180)
    f_y = f_x
    
    #Estimated camera matrix values.
    camera_matrix = np.float32(  [ [f_x, 0.0, c_x],
                                   [0.0, f_y, c_y], 
                                   [0.0, 0.0, 1.0] ])
    
#    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
    
    #Distortion coefficients
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    
    return camera_matrix, camera_distortion



def _initRefPoints():    
    #Antropometric constant values of the human head. 
    #Found on wikipedia and on:
    # "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
    #
    #X-Y-Z with X pointing forward and Y on the left.
    #The X-Y-Z coordinates used are like the standard
    # coordinates of ROS (robotic operative system)
    P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
    P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
    P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
    P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
    P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
    P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
    P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
    P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27
    P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
    P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
    P3D_RIGHT_EYE = np.float32([-20.0, -65.5,-5.0]) #36
    P3D_RIGHT_TEAR = np.float32([-10.0, -40.5,-5.0]) #39
    P3D_LEFT_TEAR = np.float32([-10.0, 40.5,-5.0]) #42
    P3D_LEFT_EYE = np.float32([-20.0, 65.5,-5.0]) #45
    #P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
    #P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
    P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62
    
    #The points to track
    #These points are the ones used by PnP
    # to estimate the 3D pose of the face
    TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
    ALL_POINTS = list(range(0,68)) #Used for debug only
    
    #This matrix contains the 3D points of the
    # 11 landmarks we want to find. It has been
    # obtained from antrophometric measurement
    # on the human head.
    landmarks_3D = np.float32(   [P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])
    
    return TRACKED_POINTS, ALL_POINTS, landmarks_3D


def axis(x):
    axisPts = np.float32(       [[x,0,0], 
                              [0,x,0], 
                              [0,0,x]])
    return axisPts


def box(x, y, z):
    
#    ox, oy, oz = -x,-x,x
#    ex, ey, ez = x,x,-x
#    ox, oy, oz = -x,-x,x
    ox, oy, oz = 0,-y,-x
    ex, ey, ez = z,y,x
    boxPts = np.float32([[ox,oy,oz], [ox,ey,oz], [ex,ey,oz], [ex,oy,oz],
               [ox,oy,ez],[ox,ey,ez],[ex,ey,ez],[ex,oy,ez] ])
    return boxPts


def draw_axis(frame, landmarks_2D, imgpts, scale):
    #Drawing the three axis on the image frame.
    #The opencv colors are defined as BGR colors such as: 
    # (a, b, c) >> Blue = a, Green = b and Red = c
    #Our axis/color convention is X=R, Y=G, Z=B
    imgpts*=scale
    landmarks_2D*=scale
    sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
    cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
    cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED


def draw_box(img, imgpts, scale):
    imgpts = np.int32(imgpts).reshape(-1,2)*scale
    
    # draw ground floor in green
    cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    

#ADD TEXT
def add_text(labels_list, x, y, w, h, scale, font = cv2.FONT_HERSHEY_SIMPLEX, line_type = cv2.LINE_AA):
    x_text = int( x * scale + w*scale/2 ) 
    y_text = int(y * scale - h*scale/2 )
    for label in labels_list:
        cv2.putText(mark_up, label, (x_text, y_text), font, 0.8, (0,255,255), 2, line_type)
        y_text += 30


def init_all():
    # init my functions
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    
    FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))
    
    TRACKED_POINTS, ALL_POINTS, landmarks_3D = _initRefPoints()
    camera_matrix, camera_distortion = _initCamMatrix(cam_w = cam_w, cam_h = cam_h)
    
    box3D = box(40, 80, 80)
    axis3D = axis(40)
    
    scale = 3
    
    labels_list = "#1,Larry,Happy Chappy".split(",")
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    


# MAIN DRIVER #

video_capture = cv2.VideoCapture(0)
cv2.namedWindow('Video')
cv2.moveWindow('Video', 20, 20)

start = time.time()
frame_times=[]


# init my functions
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam_w = int(video_capture.get(3))
cam_h = int(video_capture.get(4))

TRACKED_POINTS, ALL_POINTS, landmarks_3D = _initRefPoints()
camera_matrix, camera_distortion = _initCamMatrix(cam_w = cam_w, cam_h = cam_h)

box3D = box(40, 80, 80)
axis3D = axis(40)

scale = 3

labels_list = "#1,Larry,Happy Chappy".split(",")
font = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA

## MAIN LOOP ##
while(video_capture.isOpened() == True):

    # Capture frame-by-frame
    ret, frame_in = video_capture.read()
    
    #make copies of input frames
    mark_up = frame_in.copy()
    xdim, ydim, lyrs = frame_in.shape
    #Downsample frame
    frame = cv2.resize(frame_in, (int(ydim/scale), int(xdim/scale)))
    
    faces = FACE_DETECTOR.detectMultiScale(frame, 1.3, 5, minSize=(48, 48))
#    print(len(faces))
    for i, (x,y,w,h) in enumerate(faces):
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        pos = dlib.rectangle( *(int(i) for i in [x,y,x+w,y+h]))
        
        dlib_landmarks = predictor(frame, pos)
        landmarks_2D = np.zeros((len(TRACKED_POINTS),2), dtype=np.float32)
        
        counter = 0
        for point in ALL_POINTS:
            x, y = dlib_landmarks.parts()[point].x, dlib_landmarks.parts()[point].y
            
            if point in TRACKED_POINTS:
                landmarks_2D[counter] = [x, y]
                cv2.circle(mark_up,(x * scale, y * scale), 4, (0,0,255), -1)
                counter += 1
                
            cv2.circle(mark_up,(x * scale, y * scale), 2, (0,255,0), -1)
              
        retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                          landmarks_2D, 
                                          camera_matrix, camera_distortion,
                                          cv2.SOLVEPNP_ITERATIVE)
        
        imgpts, jac = cv2.projectPoints(box3D, rvec, tvec, camera_matrix, camera_distortion)
        draw_box(mark_up, imgpts, scale)
        
        imgpts, jac = cv2.projectPoints(axis3D, rvec, tvec, camera_matrix, camera_distortion)
        draw_axis(mark_up, landmarks_2D, imgpts, scale)
        
        add_text(labels_list, x, y, w, h, scale, font=font, line_type=line_type)
            
    frame = cv2.resize(frame, (ydim, xdim))    
    alpha = 0.8
    output = cv2.addWeighted(mark_up, alpha, frame_in, 1 - alpha, 0)
    cv2.imshow('Video', output)
    
    # FPS measurements
    end = time.time() 
    delta = time.time() - start
    start = end
    frame_times.append(delta)
    if len(frame_times) > 10:
        frame_times.pop(0)
    fps = len(frame_times) / sum(frame_times)
    print(fps)
    
    
    # EXIT MAIN LOOP
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
for i in range(5):
    cv2.waitKey(1)