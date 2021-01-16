from __future__ import print_function
import cv2 as cv
import math


def detect(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    cv.imshow('grey', frame_gray)
    cv.imshow('',cv.resize(frame_gray,(64,64),0))
    
    faces = face_cascade.detectMultiScale(frame_gray)

    print(len(faces),' face detected')

    return faces

def display(frame,faces):
    print(faces)

    for i in range(1,8):
            frame = cv.line(frame,(0,(40*i)),(320,(40*i)),(255,255,255))
            frame = cv.line(frame,((40*i),0),((40*i),320),(255,255,255))
    
    for (x,y,w,h) in faces:  
        # center = (x + w//2, y + h//2)
        p1 = (x,y)
        p2 = (x+w,y+h)
        # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        frame = cv.rectangle(frame,p1,p2,(255,255,255),thickness=2)
        # faceROI = frame_gray[y:y+h,x:x+w]
        # cv.imshow('ROI',faceROI)
        
    cv.imshow('Capture - Face detection', frame)
    
def sampling (faces):
    for (x,y,w,h) in faces:
        target = (round((x + w/2)/40), round((y + h/2)/40))
        print('x : ',target[0],' | y : ',target[1])
        


face_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile('./face.xml')):
    print("error - face weight")
    exit(-2)


cap = cv.VideoCapture(-1)
cap.set(3, 320)
cap.set(4, 320)
cap.set(5, 30)


if not cap.isOpened:
    print("error - camera")
    exit(-1)

while True:
    ret, frame = cap.read()
    if frame is None:
        print("error - no frame")
        break

    faces = detect(frame)
    display(frame,faces)
    sampling(faces)

    ## imshow 필수

    if cv.waitKey(10)==27:
        break
