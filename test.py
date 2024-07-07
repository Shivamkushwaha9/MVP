import cv2
import streamlit as st

st.title("OpenCV Camera Feed in Streamlit")
image_placeholder = st.empty()
# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # 0 for default camera

def cvv():
    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture image")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_placeholder.image(frame, channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Streamlit image placeholder
image_placeholder = st.empty()

#Calling the function of MAIN CV CODEBASE
func=cvv()
    
cap.release()
cv2.destroyAllWindows()

