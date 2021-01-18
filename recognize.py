from __future__ import print_function
import cv2 as cv
import numpy as np

def detect(frame):
    
    data = []

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray) 
    faces = face_cascade.detectMultiScale(frame_gray)
    print(len(faces),' face detected')

    # 인식된 얼굴 위치에서 마스크를 검출하기 위한 부분 선택
    for (x,y,w,h) in faces:
        
        #관심부분을 선택
        maskROI = frame_gray[y+(h*2//3):y+h,x:x+w]
        # cv.imshow('mask',maskROI)
        
        # 입을 인식해 마스크 착용 여부 확인 ( 1 = 착용,  0 = 미착용 )
        mask = mouth_cascade.detectMultiScale(maskROI)
        print(type(mask))
        if type(mask) is tuple:
            mask = 1
        else:
            mask = 0
        data.append([x,y,w,h,mask])
        print(mask)

    # cv.imshow('mouth',frame)    

    data = np.array(data)

    return data



#def collectiROI(frame):
    # gathering ROI Data ( identify face )
    # faceROI = frame_gray[y:y+h,x:x+w]
    # cv.imshow('ROI',faceROI)
         

def get_data(faces,thermo_data):
    
    data = []

    for(x,y,w,h,mask) in faces:
        # 이마 위치를 target으로 설정
        target = [round((x+ w//2)/40), round((y + h//6)/40)]
        thermo = thermo_data[int(target[0]-1)][int(target[1]-1)]
        data.append([x,y,w,h,mask,thermo])
    
    data = np.array(data)
    
    return data




def display(frame,data):

    #  8*8 격자
    for i in range(1,8):
            frame = cv.line(frame,(0,(40*i)),(320,(40*i)),(255,255,255))
            frame = cv.line(frame,((40*i),0),((40*i),320),(255,255,255))
    
    for (x,y,w,h,mask,thermo) in data:  
        
        # 이마 위치 표시
        # forehead_1 = (int(x + w//3),int(y + 0))
        # forehead_2 = (int(x + 2*(w//3)), int(y + 1*(h//3))) 
        # frame = cv.rectangle(frame,forehead_1,forehead_2,(0,0,255),thickness=2)

        p1 = (int(x),int(y))
        p2 = (int(x+w),int(y+h))

        print(mask)

        if mask == 0 :
            frame = cv.rectangle(frame,p1,p2,(0,0,255),thickness=3)


            frame = cv.putText(frame,'No Mask',p1,0,1,(255,255,255),thickness=2)

        else :
            frame = cv.rectangle(frame,p1,p2,(255,255,255),thickness=2)


            frame = cv.putText(frame,str(thermo),p1,0,1,(255,255,255),thickness=2)
        
    cv.imshow('Capture - Face detection', frame)

face_cascade = cv.CascadeClassifier()
mouth_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile('./face.xml')):
    print("Error - face data not found")
    exit(-2)

if not mouth_cascade.load(cv.samples.findFile('./mouth.xml')):
    print("Error - mouth data not found")
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
    data = get_data(faces,thermo_data)
    

    # display realtime video & information
    display(frame,data)

    ## imshow 필수

    if cv.waitKey(10)==27:
        break
