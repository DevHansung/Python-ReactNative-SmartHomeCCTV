소개
============
#### Python 얼굴인식 딥러닝 + Raspberrypi + ReactNative로 구현한 SmartHomeCCTV
#### 실행영상: https://youtu.be/XwGct3jzOe4


설명
====
### 1. 개발환경:  
* Raspberrypi(Raspbian): Python Flask   
* pc(Ubuntu): Python Flask   
* Application: ReactNative   
* 기타 라이브러리: OpenCV   

### 2. 개발인원  
* 3명   
### 3. 개발기간  
* 1달   
### 4. 본인의 프로젝트 기여도
* 40%   
### 5. 본인이 맡은 역할
* 라즈베리파이에서 실행되거나 직접적으로 연관된 기능 개발   
 (Flask API서버 개발, 라즈베리파이 카메라와 연동한 Python얼굴인식 알고리즘 구현 등)   

### 6. 동작 설명  
 * 라즈베리파이에서 영상 촬영을 하고 실시간으로 스트리밍 서버에 올린 뒤  
 PC로 연결해서 스트리밍 서버에 접속하여 실시간으로 영상을 분석하고 결과를 앱으로 전송한다.  
 앱에서는 얼굴인식 결과를 확인 할 수 있으며 <br>
 방향키 버튼을 배치해 서보모터를 상,하,좌,우로 동작하여 카메라 각도를 조절하고,  
 영상을 켜고 끄는 등 라즈베리파이를 원격으로 제어 할 수 있다.<br>
 또한 사용자의 안면을 등록하여 스트리밍 되는 영상의 안면과 딥러닝 기법으로 대조, 분석하여  
 검출률이 80%미만(테스트 결과 90~80%선에서 본인이 아니라 판단)이면 <br>
 무단침입자로 간주하여 즉시 경고 알림을 띄우고 알림을 클릭하면 비상연락처로 연결한다.  
 (PC에 우분투를 설치하고 실시간 스트리밍되는 영상에 대하여 캡쳐, 분석, 검출 실행  
 라즈베리파이로 직접 검출도 가능하지만 기기 사양이 낮아 속도 저하(검출률 저하))  

### 7. 실행 방법:  
* 서버측 사전 작업  
   1.라즈베리파이로 영상 촬영 및 스트리밍 켜기  
   2.api.py를 터미널로 실행(라즈베리파이에서 실행)  
   3.pc서버로 라즈베리파이 서보 블라스터 원격 실행  
   4.pc서버로 얼굴인식 촬영, 훈련, 검출 원격 실행  

### * 기능 동작  
   1.RestAPI방식의 url로 기능 컨트롤 가능  
     어플리케이션에서는 배치된 버튼에 해당 기능의 url값을 줘 버튼을 클릭하면 해당 기능이 수행된다.  

        실행ex)  
        curl http://192.168.0.0:5000/leftpan/1  
        curl http://192.168.0.0:5000/dataset/1  
        버튼 클릭 형식으로 실행하면 위와같은 url 요청으로 실행 하는것과 동일한 동작   

### 8. 특이사항  
 본 프로젝트는 예전에 완성을 한 뒤 개인적인 저장 매체에 보관하고 있다가 파일을 옮기고 합치는 과정에서  
 프론트엔드 부분인 reactNative로 구현한 어플리케이션 부분의 코드가 분실 되었습니다.  
 없어진 부분과 관계없이 파이썬 얼굴인식 코드는 라즈베리파이와 카메라만 있으면  
 기능적 면에서 문제없이 테스트와 실행이 가능하고 결과도 낼 수 있어 이부분만 올리도록 하였습니다.  
 
기타 참고사항 
=============
### 동작 설명은 실행 영상 참조-> https://youtu.be/XwGct3jzOe4 
