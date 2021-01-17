from __future__ import print_function
import cv2 as cv
import numpy as np

def detect(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray) 
    faces = face_cascade.detectMultiScale(frame_gray)
    print(len(faces),' face detected')

    return faces

def display(frame,data):

    for i in range(1,8):
            frame = cv.line(frame,(0,(40*i)),(320,(40*i)),(255,255,255))
            frame = cv.line(frame,((40*i),0),((40*i),320),(255,255,255))
    
    for (x,y,w,h,thermo_data) in data:  
        
        forehead_1 = (int(x + w//3),int(y + 0))
        
        forehead_2 = (int(x + 2*(w//3)), int(y + 1*(h//3)))
        
        frame = cv.rectangle(frame,forehead_1,forehead_2,(0,0,255),thickness=2)
        
        # forehead_11 = (x + 5*(w//12), y + 1*(h//12))
        # forehead_21 = (x + 7*(w//12), y + 3*(h//12))
        # frame = cv.rectangle(frame,forehead_11,forehead_21,(0,255,0),thickness=2)
        
        p1 = (int(x),int(y))
        p2 = (int(x+w),int(y+h))
        frame = cv.rectangle(frame,p1,p2,(255,255,255),thickness=2)

        frame = cv.putText(frame,str(thermo_data),p1,0,1,(255,255,255),thickness=2)

        
        

        
     
    cv.imshow('Capture - Face detection', frame)

#def collectiROI(frame):
    # gathering ROI Data ( identify face )
    # faceROI = frame_gray[y:y+h,x:x+w]
    # cv.imshow('ROI',faceROI)
         
def targeting (faces):
    target = []
    print(faces)
    for(x,y,w,h) in faces:
        print('[',x,y,w,h,']')
        target_tmp = [round((x+ w//2)/40), round((y + h//6)/40)]
        print('origin data')
        print('x : ',x + w//2,' | y : ',y + h//6)
        print('scaled data')
        print('x : ',target_tmp[0],' | y : ',target_tmp[1])
        target.append([x,y,w,h,target_tmp[0],target_tmp[1]])

    print(target)
    target = np.array(target)
    print(target)

    return target


def get_thermo(target,thermo_data):
    data = []
    for (x,y,w,h,x1,x2) in target:
        thermo = thermo_data[int(x1-1)][int(x2-1)]     
        data.append([x,y,w,h,thermo])
    
    data = np.array(data)
    print(data)
    return data


face_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile('./face.xml')):
    print("error - face weight")
    exit(-2)


cap = cv.VideoCapture(-1)
cap.set(3, 320)
cap.set(4, 320)
cap.set(5, 30)

thermo_data = [[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8],[2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8],[3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8],[4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8],[5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8],[6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8],[7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8],[8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8]]


if not cap.isOpened:
    print("error - camera")
    exit(-1)

while True:
    ret, frame = cap.read()
    if frame is None:
        print("error - no frame")
        break

    # get information
    faces = detect(frame)
    target = targeting(faces)
    data = get_thermo(target,thermo_data)  
    

    # display realtime video
    display(frame,data)

    ## imshow 필수

    if cv.waitKey(10)==27:
        break
