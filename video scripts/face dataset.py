# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:35:02 2021

@author: HP
"""
import cv2
import os
import numpy as np

videospath=np.array(os.listdir('../test videos'))
# face_id = 0
face_id = input('\n enter user id end press <return> ==>  ')
for i in videospath:
    print(i)
    cam = cv2.VideoCapture('../test videos/'+ str(i))
    # face_id+=1
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    face_detector = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')
    
    # For each person, enter one numeric face id
    

    
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    
    createuserDir = '../video dataset/''user'+ str(face_id)
    try:    
        os.makedirs(createuserDir)
        print('directory',createuserDir,'created')
    except FileExistsError:
        print('directory',createuserDir,'already exists')
    while(True):
    
        ret, img = cam.read()
        if ret == False:
            break
        else:
            #img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("../video dataset/"+createuserDir+"/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('face dataset creator', img)
                print(x, y, w, h)
    
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 113 or k==27:
                break
            elif count >= 3000: # Take 30 face sample and stop video
                 break
    
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    print('code completed')