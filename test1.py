import cv2
import os
import numpy as np
import face_recognition as fr                                                                                           # importing the 'face_recognition' here to be able to use


test_img = cv2.imread('C:/Users/Ameya Shukla/Desktop/Python/Projects/Face Recognition using OpenCV/Images/ameya6.JPG')  # load a test image

faces_detected,gray_img = fr.faceDetection(test_img)                                                                    # we get 'faces_detected' and 'gray_img' using 'faceDetection'
                                                                                                                        # function from the previous file 'fr' (face_recognition)

print('faces_detected : ', faces_detected)                                                                              # prints the number of faces detected


'''
for (x, y, h, w) in faces_detected:                                                                                     # measurement of rectangle around the face/faces detected
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=4)


resized_img = cv2.resize(test_img, (1000,700))                                                                          # resize to see the complete image properly on the screen.
                                                                                                                        # Because the haar cascade cascade classifier is trained for
                                                                                                                        # a specific size

cv2.imshow('Face detection', resized_img)                                                                               # Title

cv2.waitKey(0)                                                                                                          # the screen waits on the screen until any key is pressed
cv2.destroyAllWindows()
'''


# For training new images
'''
faces, faceID = fr.labels_for_training_data('C:/Users/Ameya Shukla/Desktop/Python/Projects'
                                            '/Face Recognition using OpenCV/Training Images')

face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save('training_data.yml')'''

# For loading the already trained images
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/Ameya Shukla/Desktop/Python/Projects/Face Recognition using OpenCV/training_data.yml')

# name labels
name = {0:'Keanu Reeves', 1:'Ameya'}


# For multiple faces in a given image
for face in faces_detected:
    (x, y, w, h) = face                                                                                                 # reactangular part which is the face part
    roi_gray = gray_img[y:y+h, x:x+h]                                                                                   # Extracting the gray from the gray image
    label, confidence = face_recognizer.predict(roi_gray)                                                               # the 'predict' method in openCV will return the label for the image and also confidence level in the prediction
    print('label:', label)
    print('confidence:', confidence)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if (confidence > 37):
        continue
    fr.put_text(test_img, predicted_name, x, y)



# Display the image
resized_img = cv2.resize(test_img, (1000,700))                                                                          # resize to see the complete image properly on the screen. Because the haar cascade cascade classifier is trained for a specific size
cv2.imshow('Face detection', resized_img)                                                                               # Title
cv2.waitKey(0)                                                                                                          # the screen waits on the screen until any key is pressed
cv2.destroyAllWindows()


