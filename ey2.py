import cv2
import numpy as np
import time
from scipy.spatial import distance as dist

# Load the pre-trained Haar Cascade classifier for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables for counting eye movements and tracking time
look_away_count = 0
blink_count = 0
eyes_detected = False
start_time = time.time()
duration = 10 # Run for 60 seconds

# Eye aspect ratio to check for blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

# Function to compute the eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect eyes and count look-aways and blinks
def detect_eyes(frame):
    global look_away_count, eyes_detected, COUNTER, blink_count
    
    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for the face (to detect eyes)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # If eyes are detected, reset eyes_detected flag
        if len(eyes) > 0:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                eye = roi_gray[ey:ey+eh, ex:ex+ew]
                ear = eye_aspect_ratio(eye)
                
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        blink_count += 1
                    COUNTER = 0
        
        # Draw rectangles around detected eyes for visualization
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # If no eyes detected for a frame, increment look-away count
    if not eyes_detected:
        look_away_count += 1
        eyes_detected = False
    
    return frame

# Main function to capture video from webcam and process each frame
def main():
    global look_away_count
    cap = cv2.VideoCapture(0)  # Open default webcam
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_eyes(frame)
        
        # Add the timer on the side of the camera feed
        timer_text = f'Time: {int(elapsed_time)}s'
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the frame with detections and timer
        cv2.imshow('Eye Tracker with Timer', frame)
        
        # Check for 'q' key to exit loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print the count of look-aways and blinks detected
    print(f"Total look-aways detected: {look_away_count}")
    print(f"Total blinks detected: {blink_count}")

if __name__ == "__main__":
    main()