import cv2

# Load the cascade ( which means this loaded cascade is the main set of rules in order to be detected as a face 
# which is also called as multiexpert system )
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
# To capture video from webcam we need to use cv2.videoCapture(0). 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Reads the frame
    flag,facedetection = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(facedetection, cv2.COLOR_BGR2GRAY)

    '''
    Detect the faces and the parameters in the detectMultiScale are image (matrix where 
    required objects are detected) , scale factor( it is a factor upto which the original 
    frame is reduced to inorder to increase the speed of detection ) and minNeighbours 
    ( this value specifies the number of neighbour rectangles a candidate rectangle should
     have in order to get retained
    '''
    faces = face_cascade.detectMultiScale(gray, 1.1 , 15)

    eyes = eye_cascade.detectMultiScale(gray, 1.1 , 35)

    mouth = mouth_cascade.detectMultiScale(gray, 1.1 , 25)

    nose = nose_cascade.detectMultiScale(gray, 1.1 , 20)

    #here we are using len() is the built-in function that returns the number of elements in a list or the number of characters in a string.
    dummy = len(mouth)

    dummy1 = len(nose)

    if(dummy or dummy1):

        for (sx, sy, sw, sh) in eyes:

            cv2.rectangle(facedetection, (sx, sy), (sw+sx, sh+sy), (255, 0, 0), 2)

            cv2.putText(facedetection, 'eyes', (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

        # Draw the rectangle around each face by using sliding window algo
        for (x, y, w, h) in faces:

            cv2.rectangle(facedetection, (x, y), (x+w, y+h), (200, 0, 0), 2)

            cv2.putText(facedetection, 'Face', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 1)
        
        cv2.putText(facedetection, 'Without Facemask', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 2)
        
        
    else:
        for (sx, sy, sw, sh) in eyes:

            cv2.rectangle(facedetection, (sx, sy), (sw+sx, sh+sy), (255, 0, 0), 2)

            cv2.putText(facedetection, 'eyes', (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

        # Draw the rectangle around each face by using sliding window algo
        for (x, y, w, h) in faces:

            cv2.rectangle(facedetection, (x, y), (x+w, y+h), (200, 0, 0), 2)

            cv2.putText(facedetection, 'Face', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 2)
        
        cv2.putText(facedetection, 'With Facemask', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
        


    # Display here imshow is the default function which shows the resultant frame or capture
    
    cv2.imshow('Face Detection', facedetection)
    

    # Stop if escape key is pressed, here we are using a value of 50 ms which means we are reading the frames at a speed of 20 frames per second
   
    k = cv2.waitKey(50) 

    # here we are using k==27 because we need to stop the webcam using escape key ( whose default value is 27 )

    if k==27:
        break
        
# Release the VideoCapture object after pressing the escape key
cap.release()