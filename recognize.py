from __future__ import print_function
import cv2 as cv
import numpy as np

camsize = 320
scale = int(camsize/8)
DB= []


def detect(frame):
    
    data = []

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray) 
    faces = face_cascade.detectMultiScale(frame_gray)
    print(len(faces),' face detected')

    # 인식된 얼굴 위치에서 마스크를 검출하기 위한 부분 선택
    for (x,y,w,h) in faces:
           
        #마스크 착용여부 판단을 위한 관심부분 선택
        maskROI = frame_gray[y+(h*2//3):y+h,x:x+w]
        
        # 입을 인식해 마스크 착용 여부 확인 ( 1 = 착용,  0 = 미착용 )
        mask = mouth_cascade.detectMultiScale(maskROI,scaleFactor=2,minNeighbors = 3)
        if type(mask) is tuple:
            mask = 1
        else:
            mask = 0



        data.append([x,y,w,h,mask])

    data = np.array(data)

    return data, frame


# def collectFace(frame,data):
#     for(x,y,w,h,mask) in data:
#         number = np.where(data==x)[0][0]

#         faceROI = frame[y:y+h,x:x+w]
#         faceROI = cv.cvtColor(faceROI,cv.COLOR_BGR2GRAY)
#         faceROI = cv.equalizeHist(faceROI)
#         cv.imwrite("/home/pi/Desktop/test/haar_cascade/Data/userface"+str(number)+"_"+str(x)+".jpg",faceROI )


def get_data(faces,thermo_data):
    
    data = []

    for(x,y,w,h,mask) in faces:
    # for(x,y,w,h,mask,name) in faces:
        # 이마 위치를 target으로 설정
        target = [round((int(x)+ int(w)//2)/scale), round((int(y) + int(h)//6)/scale)]
        thermo = thermo_data[int(target[1]-1)][int(target[0]-1)]
        
        data.append([x,y,w,h,mask,thermo])
        # data.append([x,y,w,h,mask,name,thermo])
    
    data = np.array(data)
    
    return data




def display(frame,data):

    #  8*8 격자
    for i in range(1,8):
            frame = cv.line(frame,(0,(scale*i)),(camsize,(scale*i)),(255,255,255))
            frame = cv.line(frame,((scale*i),0),((scale*i),camsize),(255,255,255))
    
    for (x,y,w,h,mask,thermo) in data:  
    # for (x,y,w,h,mask,name,thermo) in data:  
        
        
        p1 = (int(x),int(y-15))
        p2 = (int(x+w),int(y+h))
        p3 = (int(x),int(y+h+25))
        p4 = (int(x+60),int(y+h+25))

        

        if mask == 0 :
            frame = cv.rectangle(frame,p1,p2,(0,0,255),thickness=3)
            frame = cv.rectangle(frame,p3,p4,(0,0,0),thickness=-1)
            frame = cv.putText(frame,'No Mask',p1,0,1,(255,255,255),thickness=2)
            frame = cv.putText(frame,str(thermo),p3,0,1,(255,255,255),thickness=2)
            # frame = cv.putText(frame,str(name),p2,0,1,(255,255,255),thickness=2)
            
        else:
            frame = cv.rectangle(frame,p1,p2,(255,255,255),thickness=2)
            frame = cv.putText(frame,str(thermo),p3,0,1,(255,255,255),thickness=2)
            # frame = cv.putText(frame,str(name),p2,0,1,(255,255,255),thickness=2)
            # if (float(thermo) >= 37.5 or float(thermo) <= 35.5) :
            #     frame = cv.putText(frame,"!!Need Check!!",p1,0,1,(0,0,255),thickness=2)
                  
    cv.imshow('Capture - Face detection', frame)
    cv.moveWindow('Capture - Face detection',0,0)
    





face_cascade = cv.CascadeClassifier()
mouth_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile('/home/pi/Desktop/test/haar_cascade/face.xml')):
    print("Error - face data not found")
    exit(-2)

if not mouth_cascade.load(cv.samples.findFile('/home/pi/Desktop/test/haar_cascade/mouth.xml')):
    print("Error - mouth data not found")
    exit(-2)

cap = cv.VideoCapture(-1)

cap.set(3, camsize)
cap.set(4, camsize)
cap.set(5, 30)

thermo_data = [[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8],[2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8],[3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8],[4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8],[5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8],[6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8],[7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8],[8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8]]
# thermo_data = [[37.7,34.7,34,34.1,37.8,37.2,36.3,37.4],
# [36.5,36.5,36.5,36.5,36.5,36.5,36.5,36.5],
# [36.5,36.5,36.5,36.5,36.5,36.5,36.5,36.5],
# [36.5,36.5,36.5,36.5,36.5,36.5,36.5,36.5],
# [37.9,37.9,37.9,37.9,37.9,37.9,37.9,37.9],
# [37.9,37.9,37.9,37.9,37.9,37.9,37.9,37.9],
# [37.9,37.9,37.9,37.9,37.9,37.9,37.9,37.9],
# [34.4,34.5,36.4,35,37.8,34.5,36.5,37.7]
# ]

if not cap.isOpened:
    print("error - camera")
    exit(-1)

while True:
    ret, frame = cap.read()
    
    if frame is None:
        print("error - no frame")
        break

    # get information
    faces, frame_gray = detect(frame) 
    data = get_data(faces,thermo_data)
    # collectFace(frame,faces)
    

    # display realtime video & information
    display(frame,data)

    ## imshow 필수

    if cv.waitKey(10)==27:
        break
