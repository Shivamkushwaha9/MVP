import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\SHIVAM\Desktop\MVP\GazeTracking\Landmark\gaze_tracking_mw.dat')

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Get the coordinates of the left and right eye landmarks
        left_eye = [landmarks.part(36).x, landmarks.part(36).y,
                    landmarks.part(39).x, landmarks.part(39).y]
        right_eye = [landmarks.part(42).x, landmarks.part(42).y,
                     landmarks.part(45).x, landmarks.part(45).y]

        # Calculate the eye aspect ratio
        left_eye_center = np.array([(left_eye[0] + left_eye[2]) / 2, (left_eye[1] + left_eye[3]) / 2])
        right_eye_center = np.array([(right_eye[0] + right_eye[2]) / 2, (right_eye[1] + right_eye[3]) / 2])

        # Calculate the vector between the eye centers
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

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()