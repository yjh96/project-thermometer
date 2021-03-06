# 얼굴인식 체온측정기

Hardware

> Raspberry Pi 4B+  
> PiCamera Ver 1.3  
> Panasonic AMG8833

Software

> Opencv - 4.5.0  
> Opencv_contrib - 4.5.0  
> Python - 3.7.3  
> adafruit-circuitpython-amg88xx - 1.2.5  
> numpy - 1.16.2

-   프로그램 실행 절차

    ```
    1. Camera를 통한 얼굴 인식 [ Opencv - haar cascade]
    2. 얼굴의 위치를 열화상센서 배열과 대응대도록 연산
    3. 얼굴 위치의 [i,j] 와 대응되는 Sensor값 읽기
    4. 마스크 착용 여부 표시
    5. 35.5℃ 미만, 37.5℃ 초과 시 추가 안내 알림 표시
    ```

-   개발 기록

    ```
    2021-01-14
    - Raspberry Pi opencv & dependency 설치

    2021-01-15
    - Build opencv W/opencv-contrib
    - opencv 설치
    - opencv with PiCamera 작동 테스트

    2021-01-16
    - Haar-Cascade 샘플코드 테스트
    - 샘플코드 Refactoring 및 테스트
    - Refactoring 된 샘플코드를 바탕으로 새 코드 작성

    2021-01-17
    - Grid 생성 >> 센서 배열인 8*8에 대응하는 위치를 확인하기 위한 작업
    - Dummy Data를 통한 열화상 데이터 요청 테스트
    - 얼굴 전체가 아닌 이마의 좌표를 요청하도록 수정 ( 마스크 착용시 정확도를 위해 수정함 )
    - Target을 적색 사각형으로 표기하도록 함.

    2021-01-18
    - 여러 사람을 구분하지 못하는 문제 수정
    - Dummy Data를 화면에 표기하도록 코드 작성
    - 마스크 착용 여부를 구분하기 위해 인식된 얼굴 내부에서 입을 인식하도록 코드 작성

    2021-01-19
    - 특정 구간을 벗어나는 경우 [ 35.6 ~ 37.4 ℃ ] 알림문구 추가
    - 실제 온도와 비슷한 Dummy Data를 생성해 코드 작동 확인

    2021-01-20
    - 반복 입장하는 사람을 파악하기 위한 사진 수집 코드 작성 ( 사진 수집 기능만 구현 )

    2021-01-21
    - 실제 열화상 센서 데이터를 화면에 표시하도록 더미데이터 삭제
    - 문구 출력위치 변경 ( 화면 상단 )
    - AMG8833 센서와 I2C를 활용해 pixel의 정보를 가져오도록 코드 작성
    -  카메라 좌표와 센서 픽셀 정보가 일치하는지 확인

    2021-01-22
    - 마스크 착용 후 온도 이상시 적색 테두리 적용
    - 미사용 좌표 삭제
    ```

-   문제점

    ```
    - 실시간 인식을 위해서는 해상도를 포기해야 함 (320*320)
    --> 먼 거리에서는 마스크 착용 여부를 구분하지 못함 (해상도와 연관)
    - 사진 수집 알고리즘의 구현이 필요함 ( 여러 사람이 들어오는 경우 사진수집시 뒤섞이는 문제가 발생함 )
    - 열화상 센서와 거리가 멀수록 온도가 낮게 측정되는 문제 발생 ( Datasheet 상 7m까지 사용이 가능하다고 표기되어 있음 )
    ```

-   Reference

    > <a href="https://opencv.org">OpenCV</a>  
    > <a href="https://numpy.org/">Numpy</a>  
    > <a href="https://github.com/adafruit/dafruit_CircuitPython_AMG88xx">Adafruit-AMG88xx</a>

-   Package

```
# Adafruit-AMG88xx
$sudo pip3 install adafruit-circuitpython-amg88xx
```
