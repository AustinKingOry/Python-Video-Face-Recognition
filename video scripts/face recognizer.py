# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:38:26 2021

@author: HP
"""

import cv2
import numpy as np
import os 
from random import randrange
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../trainer/trainer.yml')
cascadePath = "../cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
idcounter = 0
names = []
# Initialize and start realtime video capture
videospath=os.listdir('../test videos')
print(videospath)
for cl in videospath:
    names.append(os.path.splitext(cl)[0])
print(names)
face_id = 0
for i in videospath:
    print(i)
    cam = cv2.VideoCapture('../test videos/'+ str(i))
    face_id+=1
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 2,
            minSize = (int(minW), int(minH)),
           )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(128,256),randrange(128,256),randrange(128,256)), 2)
            cv2.rectangle(img, (x,y-45), (x+w,y), (40,180,0), cv2.FILLED)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # If confidence is less them 100 ==> "0" : perfect match 
            if (confidence < 100):
                id = names[idcounter]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(
                        img, 
                        str(id).upper(), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                       )
            cv2.putText(
                        img, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                       )  
        
        cv2.imshow('face recognizer',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27 or k==113:
            break
    # Do a bit of cleanup
    idcounter+=1
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    print('code completed')