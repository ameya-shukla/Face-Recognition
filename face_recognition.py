import cv2
import os
import numpy as np

# 'faceDetection' takes a 'test_img' which is being already loaded using imread
def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)                                                                  # lets convert the image to gray scale image
                                                                                                                        # because our classifier takes in a gray image
                                                                                                                        # because it does not matter while detecting face.
                                                                                                                        # So, we want to remove that distracting feature
    face_haar_cascade = cv2.CascadeClassifier('C:/Users/Ameya Shukla/Desktop/Python/Projects/Face Recognition using '   # 'haar cascade classifier' are already trained to detect certain objects
                                              'OpenCV/haarcascade_frontalface_default.xml')

    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=10)                             # Returns a rectangle around where the face is detected
                                                                                                                        # 'scaleFactor=1.32' scales the image to 32% of the original
                                                                                                                        # size because 'haar cascade classifier' have been trained
                                                                                                                        # for a particular size and we need to bring the image to
                                                                                                                        # that size to be detected

                                                                                                                        # 'minNeighbors=10' is because a lot of false positives will
                                                                                                                        # be detected.So, by setting it to 10, we are saying that it
                                                                                                                        # should have atleast 10 neighbors to be detected as a
                                                                                                                        # true positive
    return faces, gray_img                                                                                              # we return the 'gray_img' for training the classifier later 'faces' will return the rectangle around the face detected






# Crop the image(Getting Training Data)
def labels_for_training_data(directory):                                                                                # labelling each taining data images
    faces = []
    faceID = []

    for path,subdir,files in os.walk(directory):                                                                        # 'os.walk' goes through each directory and file in the path
        for file in files:                                                                                              # if any file or is a system file in the path that starts
            if file.startswith('.'):                                                                                    # with a '.', we skip that file
                print('Skipping system file')
                continue

            id = os.path.basename(path)                                                                                 # image ids/labels in that path
            img_path = os.path.join(path, file)                                                                         # will need the image path later for applying to the classifier
            print('img_path:', img_path)
            print('id:', id)
            test_img = cv2.imread(img_path)                                                                             # Read the image from the given path('img_path')
            if img_path is None:                                                                                        # If image is not loaded properly, then 'imread' returns 'None'
                print('Problem while loading the image')
                continue

            faces_rect, gray_img = faceDetection(test_img)                                                              # 'faces_rect' will give us a gray image 'gray_img' by
                                                                                                                        # applying the function('faceDetection')
            if len(faces_rect) != 1:
                continue                                                                                                # Since we are assuming only single person images for training, which means 'faces_rect' should be 1

            (x, y, w, h) = faces_rect[0]                                                                                # Rectangle around the face detected. faces_rect[0] means the 1st entry in it.
            roi_gray = gray_img[y:y+w, x:x+h]                                                                           # roi_gray = region of interest in the gray image which is the rectangle i.e. the face part only
            faces.append(roi_gray)                                                                                      # appends each ractangular part i.e. each face as an entry in the list('faces') for training
            faceID.append(int(id))                                                                                      # appends labels of each ractangular part i.e. labels of each face as an entry in the list('faceID') for training
    return faces, faceID





# Train the data with classifier
def train_classifier(faces, faceID):
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()                                                              # Using the 'LBPHFaceRecognizer' classifier for tarining
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))                                                                      # training process. 'faceID' has been passed as a numpy array because 'LBPHFaceRecognizer' takes in an int value as an array
    return face_recognizer



# To Create a box around the face
def draw_rect(test_img, face):
    (x, y, w, h) = face                                                                                                 # Using the rectangular co-orodinates which we got in 'faces' object from the 'faceDetection' function above
    cv2.rectangle(test_img, (x,y), (x+w,y+h), (0,0,255), thickness=4)                                                   # Using 'rectangle' function from openCv(cv2) to create a rectangle box around the face




# To write a text of whose name it is around the box.
def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,255), 3)