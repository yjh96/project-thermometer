# 얼굴인식 체온측정기

-   List

    ```
      Hardware
      > Raspberry Pi 4B+
      > PiCamera Ver 1.3

      Software
      > Opencv-4.5.0
      > Opencv_contrib-4.5.0
      > Python-3.7.3
    ```

-   프로그램 개요

    ```
    1. Camera를 통한 얼굴 인식 [ Opencv - haar cascade]
    2. 얼굴의 위치를 열화상센서 배열과 대응대도록 연산
    3. 얼굴 위치의 [i,j] 와 대응되는 Sensor값 읽기
    4. 마스크 착용 여부 표시
    5. 35.5℃ 미만, 37.5℃ 초과 시 추가 안내 알림 표시
    ```

-   Reference

    <a href = 'https://github.com/ageitgey/face_recognition'> Face recognition
