from __future__ import print_function
import cv2 as cv
import numpy as np
import board
import busio
import adafruit_amg88xx as ts


# cam setting
camsize = 320
scale = int(camsize/8)

# AMG8833 setting
i2c_bus = busio.I2C(board.SCL,board.SDA)
thermo = ts.AMG88XX(i2c_bus)


# 얼굴 검출 함수
def detect(frame):
    
    data = []

    #검출을 위해 그레이스케일 변환과 히스토그램 균일화 실시
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray) 
    faces = face_cascade.detectMultiScale(frame_gray)
    print(len(faces),' face detected')

    # 인식된 얼굴 위치에서 마스크를 검출하기 위한 부분 선택
    for (x,y,w,h) in faces:
           
        #마스크 착용여부 판단을 위한 관심부분 선택
        mask_roi = frame_gray[y+(h*2//3):y+h,x:x+w]
        
        # 입을 인식해 마스크 착용 여부 확인 ( 1 = 착용,  0 = 미착용 )
        mask = mouth_cascade.detectMultiScale(mask_roi)

        # mask에 정보가 없을 경우 type이 tuple,
        # 정보가 입력될 경우 numpy.ndarray 이므로
        # type을 구분하도록 설정
        if type(mask) is tuple:
            mask = 1
        else:
            mask = 0

        # 얼굴 좌표에 마스크 정보를 포함해 리스트에 추가하도록 함.
        data.append([x,y,w,h,mask])

    # numpy.ndarray 형식으로 변환
    data = np.array(data)

    return data

# # 영상 중 인식된 얼굴부분을 jpg로 저장함.
# def collect_face(frame,data):
#     for(x,y,w,h,mask) in data:
#         number = np.where(data==x)[0][0]
#         # 관심부분 설정
#         face_roi = frame[y:y+h,x:x+w]
#         face_roi = cv.cvtColor(face_roi,cv.COLOR_BGR2GRAY)
#         face_roi = cv.equalizeHist(face_roi)
#         cv.imwrite("./Data/userface"+str(number)+"_"+str(x)+".jpg",face_roi )

# 인식된 얼굴과 대응되는 열화상 정보 획득
def get_data(faces,thermo_data):
    
    data = []

    for(x,y,w,h,mask) in faces:
    
    # 신원 구분이 가능한 경우 사용
    # for(x,y,w,h,mask,name) in faces: 
       
        # 이마 위치를 target으로 설정
        target = [round((int(x)+ int(w)//2)/scale), round((int(y) + int(h)//6)/scale)]
        thermo = thermo_data[int(target[1]-1)][int(target[0]-1)]
       
        # 인식된 얼굴의 좌표, 마스크 착용여부, 온도정보를 리스트에 추가
        data.append([x,y,w,h,mask,thermo])
        
        # 신원 구분이 가능한 경우 사용
        # data.append([x,y,w,h,mask,name,thermo])

    data = np.array(data)
    return data

def i2c_thermo(thermo):

    data = thermo.pixels
    data.reverse()
    return data


# 화면 출력
def display(frame,data):

    #  8*8 격자 
    for i in range(1,8):
            frame = cv.line(frame,(0,(scale*i)),(camsize,(scale*i)),(255,255,255))
            frame = cv.line(frame,((scale*i),0),((scale*i),camsize),(255,255,255))
    
    for (x,y,w,h,mask,thermo) in data: 
    # 신원 구분이 가능한 경우 사용
    # for (x,y,w,h,mask,name,thermo) in data:  
        

        # BGR 색상
        red = (0,0,255)
        blue = (255,0,0)
        green = (0,255,0)
        white = (255,255,255)
        black = (0,0,0)

        # 출력 문구
        mask_alert = "!! Wear MASK !!"
        thermo_pass = "PASS"
        thermo_alert = "!! Thermo Alert !!"

        # 출력문구 길이 계산
        mask_alert_len = cv.getTextSize(mask_alert,fontFace=0,fontScale=1,thickness=2)
        thermo_pass_len = cv.getTextSize(thermo_pass,fontFace=0,fontScale=1,thickness=2)
        thermo_alert_len = cv.getTextSize(thermo_alert,fontFace=0,fontScale=1,thickness=2)
        
        # 좌표
        # p1, p2 => 얼굴 위치 좌상단, 우측하단 표기
        p1 = (int(x),int(y))
        p2 = (int(x+w),int(y+h))
        p3 = (int(x),int(y+h+40))
        p4 = (int(x+80),int(y+h+5))
        p5 = (0,0) 
        p6 = (320,40)
        
        


        if mask == 0 :
            frame = cv.rectangle(frame,p1,p2,red,thickness=3)
            frame = cv.rectangle(frame,p5,p6,black,thickness=-1)
            frame = cv.putText(frame,mask_alert,(int((camsize-mask_alert_len[0][0])/2),int(mask_alert_len[0][1])),0,1,red,thickness=2)
            frame = cv.putText(frame,str(thermo),p3,0,1,white,thickness=2)
            
        else:
            frame = cv.rectangle(frame,p1,p2,white,thickness=2)
            frame = cv.rectangle(frame,p5,p6,black,thickness=-1)
            frame = cv.putText(frame,thermo_pass,(int((camsize-thermo_pass_len[0][0])/2),int(thermo_pass_len[0][1])),0,1,green,thickness=2)
            frame = cv.putText(frame,str(thermo),p3,0,1,white,thickness=2)
            
            # 이상온도 구분 [ 정상온도 범위 35.6 ~ 37.4 ]
        if (float(thermo) >= 37.5 or float(thermo) <= 35.5) :
                frame = cv.rectangle(frame,p5,p6,black,thickness=-1)
                frame = cv.putText(frame,thermo_alert,(int((camsize-thermo_alert_len[0][0])/2),int(thermo_alert_len[0][1])),0,1,red,thickness=2)
                  

    cv.imshow('display', frame)
    cv.moveWindow('display',0,0)
    
    





face_cascade = cv.CascadeClassifier()
mouth_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile('./face.xml')):
    print("Error - face data not found")
    exit(-2)

if not mouth_cascade.load(cv.samples.findFile('./mouth.xml')):
    print("Error - mouth data not found")
    exit(-2)

cap = cv.VideoCapture(-1)

# cam capture setting
cap.set(3, camsize)
cap.set(4, camsize)
cap.set(5, 30)

# dummy coordinate
# thermo_data = [[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8],
# [2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8],
# [3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8],
# [4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8],
# [5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8],
# [6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8],
# [7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8],
# [8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8]]

# dummy thermo data
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

    # 정보수집
    faces = detect(frame)
    thermo_data = i2c_thermo(thermo)    
    data = get_data(faces,thermo_data)

    # collect_face(frame,faces)
    
    # 실시간 영상송출 및 정보 표시
    display(frame,data)

    # ESC 입력시 프로그램 종료
    if cv.waitKey(10)==27:
        break
