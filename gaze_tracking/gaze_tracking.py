from __future__ import division
import os
import cv2
import dlib
from eye import Eye
from calibration import Calibration
import numpy as np
import pickle
import face_recognition
import cvzone

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, r"C:\Users\SHIVAM\Desktop\MVP\GazeTracking\Landmark\shape_predictor_68_face_landmarks_GTX.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

#Face recognition Module
#loading the encode file 
file = open("C:/Users/SHIVAM/Desktop/MVP/GazeTracking/Face_rec/encodelist.p", "rb")
encodelist_wids = pickle.load(file)
file.close()
encodelist, studentid = encodelist_wids

#for landmark detection
# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\SHIVAM\Desktop\MVP\GazeTracking\Landmark\shape_predictor_68_face_landmarks_GTX.dat')

gaze = GazeTracking()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    #For face recognition
    face_curr_frame = face_recognition.face_locations(frame)
    encode_curr_frame = face_recognition.face_encodings(frame, face_curr_frame)
    
    for encodeface, faceloc in zip(encode_curr_frame, face_curr_frame):
        matches = face_recognition.compare_faces(encodelist, encodeface)
        dist = face_recognition.face_distance(encodelist, encodeface)
        
        matchidx = np.argmin(dist)
        
        if matches[matchidx]:
            print("Known face detected")
            
            y1, x2, y2, y1 = faceloc
            
            bbox = y1, x2, y2, y1 
            
            img = cvzone.cornerRect(frame, bbox,rt=0)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        #Cordinates for both eyes
        left_eye = [landmarks.part(36).x, landmarks.part(36).y,
                    landmarks.part(39).x, landmarks.part(39).y]
        right_eye = [landmarks.part(42).x, landmarks.part(42).y,
                    landmarks.part(45).x, landmarks.part(45).y]

        #eye aspect ratio ke liye
        left_eye_center = np.array([(left_eye[0] + left_eye[2]) / 2, (left_eye[1] + left_eye[3]) / 2])
        right_eye_center = np.array([(right_eye[0] + right_eye[2]) / 2, (right_eye[1] + right_eye[3]) / 2])
        
        #Distancw calculating between both eyes
        eye_vector = right_eye_center - left_eye_center

        # Calculate the angle of the eye vector
        eye_angle = np.arctan2(eye_vector[1], eye_vector[0])
        eye_angle_degrees = np.degrees(eye_angle)

        # Determine if the person is looking left, right, or straight
        if eye_angle_degrees < -25:
            direction = "Looking Left"
        elif eye_angle_degrees > 25:
            direction = "Looking Right"
        else:
            direction = "Looking Straight"

        # Draw the eye landmarks and direction on the frame
        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    # To exit press q
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
   
# cv_script()
cap.release()
cv2.destroyAllWindows()