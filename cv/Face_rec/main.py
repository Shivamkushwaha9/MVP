import cv2
import pickle
import numpy as np
import face_recognition
import cvzone

#loading the encode file 
file = open(r'C:\Users\SHIVAM\Desktop\MVP\GazeTracking\Face_rec\encodelist.p', "rb")
encodelist_wids = pickle.load(file)
file.close()
encodelist, studentid = encodelist_wids
#print(encodelist)


cap = cv2.VideoCapture(0)
while True:
    
    ret, img = cap.read()
    
    face_curr_frame = face_recognition.face_locations(img)
    encode_curr_frame = face_recognition.face_encodings(img, face_curr_frame)
    
    for encodeface, faceloc in zip(encode_curr_frame, face_curr_frame):
        matches = face_recognition.compare_faces(encodelist, encodeface)
        dist = face_recognition.face_distance(encodelist, encodeface)
        
        matchidx = np.argmin(dist)
        
        if matches[matchidx]:
            print("Known face detected", matchidx)
            
            y1, x2, y2, y1 = faceloc
            
            bbox = y1, x2, y2, y1 
            
            img = cvzone.cornerRect(img, bbox,rt=0)
        else:
            print("unknown face detected")
        
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)