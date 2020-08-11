# Face Recognition by webacam

import cv2
import os
import numpy as np
import face_recognition as fr

# For loading the already trained images
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/Ameya Shukla/Desktop/Python/Projects/Face Recognition using OpenCV/training_data.yml')

# name labels
name = {0:'Keanu Reeves', 1:'Ameya'}


# Webcam access
vid = cv2.VideoCapture(0)                                                                                               # accesses the webcam and captures the images for recognition

while True:
    ret, test_img = vid.read()                                                                                          # reads the images captured while webcam is on and gives a return value(ret)= True/False and the image(test_img) i.e the frame it has captured
    faces_detected, gray_img = fr.faceDetection(test_img)                                                               # passing the 'test_img' to 'faceDetection' function of the 'face_recognition' file to detect the faces

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img, (x,y), (x+w,y+h), (0,0,255), thickness=4)


    # For multiple faces
    for face in faces_detected:
        (x, y, w, h) = face                                                                                             # reactangular part which is the face part
        roi_gray = gray_img[y:y + h, x:x + h]                                                                           # Extracting the gray from the gray image
        label, confidence = face_recognizer.predict(roi_gray)                                                           # the 'predict' method in openCV will return the label for the image and also confidence level in the prediction
        print('label:', label)
        print('confidence:', confidence)
        fr.draw_rect(test_img, face)                                                                                    # using the 'draw_rect' function from 'face_recognition' file to create a box around the face detected
        predicted_name = name[label]                                                                                    # passing the 'label' to 'predicted_name'
        if (confidence < 50):                                                                                           # setting a confidence value for the accuracy of detection
            fr.put_text(test_img, predicted_name, x, y)                                                                 #  using the 'put_text' function from 'face_recognition' file to put a text around the face detected


        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Face detection', resized_img)
        if cv2.waitKey(10) == ord('q'):
            break



vid.release()
cv2.destroyAllWindows()