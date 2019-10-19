# -*- coding: utf-8 -*-

import cv2                                                                      # Importing the opencv
import NameFind

#   import the Haar cascades for face and eye ditection

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml') # Classifier "frontal-face" Haar Cascade
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml') # Classifier "eye" Haar Cascade
face_cascade_alt = cv2.CascadeClassifier('Haar/haarcascade_frontalface_alt.xml')


recognise = cv2.face.EigenFaceRecognizer_create(100,4000)  # creating EIGEN FACE RECOGNISER
recognise.read("Recogniser/trainingDataEigan.xml")                              # Load the training data

# -------------------------     START THE VIDEO FEED ------------------------------------------
cap = cv2.VideoCapture(0)                                                       # Camera object
# cap = cv2.VideoCapture('TestVid.wmv')   # Video object
ID = None
Cont = 0
maior = 0
while True:
    ret, img = cap.read()                                                       # Read the camera object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4,minNeighbors=5,minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)
    faces2 = face_cascade_alt.detectMultiScale(gray,1.3, 5)
    classificadores = [faces,faces2]                            # Convert the Camera to gray
    #for classificadores in classificadores:
    for (x, y, w, h) in faces:
            gray_face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))               # The Face is isolated and cropped
            eyes = eye_cascade.detectMultiScale(gray_face)
            for (ex, ey, ew, eh) in eyes:
                ID, conf = recognise.predict(gray_face)                              # Determine the ID of the photo
                NAME = NameFind.ID2Name(ID, conf)
                NameFind.DispID(x, y, w, h, NAME, gray)
                print(conf)
                if conf > maior:
                    maior = conf

    cv2.imshow('EigenFace Face Recognition System', gray)
    Cont +=1
    if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
cap.release()
cv2.destroyAllWindows()
