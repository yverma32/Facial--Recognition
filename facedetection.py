import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {'person_name': 1}
with open('labels.pickle','rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.5,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
        roi_gray= gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        id_,conf = recognizer.predict(roi_gray)
        if conf>=45:
         
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(img, name , (x+30,y-10), font, 1, (0,255,255), 2, cv2.LINE_AA)

            

    cv2.imshow('image', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
