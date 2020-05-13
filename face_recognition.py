import cv2
import numpy as np
import os
import requests
import time

def recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create() #haar인식기 관련
    recognizer.read('trainer/trainer.yml') # .yml 읽어오기
    cascadePath = "haarcascades/haarcascade_frontalface_default.xml" #haar인식기 관련
    faceCascade = cv2.CascadeClassifier(cascadePath); #haar인식기 관련
    font = cv2.FONT_HERSHEY_SIMPLEX

    #iniciate id counter
    id = 0

    # names related to ids: example ==> loze: id=1,  etc
    names = ['None', 'HanSung', 'JungHoo', 'SangMo', 'SeungHun','SangJun','SangIn'] #name_id

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture("http://192.168.0.102:8080/?action=stream") #영상 재생
    cam.set(3, 640) # set video widht(가로 프레임 크기)
    cam.set(4, 480) # set video height(세로 프레임 크기)

    # Define min window size to be recognized as a face (최소 창 크기 정의??)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    chkCount = 0

    while True:
        
        ret, img =cam.read() #이미지 얻기(영상에서 이미지 한장을 가져옴)
        #img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #그레이스케일로 변환
        
        faces = faceCascade.detectMultiScale(  #얼굴인식(가져온 이미지에서 얼굴 부분 찾기)
            gray, #그레이로 변환된 이미지
            scaleFactor = 1.2, #이미지 스케일(크기?)
            minNeighbors = 5, #얼굴 검출 후보들의 갯수
            minSize = (int(minW), int(minH)), #가능한 최소 객체 사이즈
            )

        for(x,y,w,h) in faces:
            chkCount
            chkCount+=1

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #얼굴 부분만가져오기(사각형 그리기)
            #img=적용할 이미지, (x,y)=사각형 상자의 꼭지점 좌표,
            #(x+w,y+h)=상자의 반대편 꼭지점 좌표, (0,255,0)=상자 테두리의 색상, 2=상자 라인의 두께

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w]) #불러온 .yml파일 이용해서 유사도 측정

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100): #100이하의 값일경우 name, 반올림한 유사도
                id = names[id] #name_id
                confidence = round(100 - confidence) #반올림한 유사도

            else:   #아닐경우 unknown, 반올림한 유사도
                id = "unknown" #unknown
                confidence = round(100 - confidence) #반올림한 유사도

            if(id == id):
                confidence+=confidence

            print("\n " + id +" confidence : "+ str(confidence)) #결과값 찍어줌

            # request
            if (chkCount == 5 0): #count 30번 누적일 경우
                print(chkCount) #테스트 코드
                confidence=confidence/50
                if(int(confidence)<=50): #30번째 유사도가 50%보다 작을경우 서버로 경고 메시지 보냄.
                    print("h") #테스트 코드
                    url = "http://192.168.0.103:3000/api/com/intruduers" #get 보낼 url
                    params = {'param': confidence } #파라미터 값
                    res = requests.get(url, params=params) #get 보냄
                chkCount = 0 # 30번 루프 돌면 count 초기화.
            
        #cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video #종료키
        if k == 27: #종료키
            break #종료

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release() #영상 재생 중지.
    cv2.destroyAllWindows()#실행시킨 창 닫음.
    return
