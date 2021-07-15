
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('D:/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
#rec.load("D:/trainingData.yml")
rec.read('D:/trainingdata.yml')
id=0
#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==2):
            id="alok"
            print(id)
        if id==1:
            id="amol"
            print(id)
        if id==3:
            id="anjali"
            print(id)
        if id==4:
            id="Gaurav"
            print(id)
        if id==5:
            id='rahul'
            print(id)
        if id==6:
            id="akshay"
            print(id)
        if id==7:
            id="prashant"
            print(id)
        #cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
