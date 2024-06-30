import cv2
import time

# Load the pre-trained Haar Cascade classifier for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables for counting time
look_away_time = 0
look_at_time = 0
eyes_detected = False
start_time = time.time()
duration = 30  # Run for 30 seconds

# Function to detect eyes and count look-aways
def detect_eyes(frame, prev_eyes_detected, start_state_time):
    global look_away_time, look_at_time
    
    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    eyes_detected = False
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for the face (to detect eyes)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # If eyes are detected, set eyes_detected flag
        if len(eyes) > 0:
            eyes_detected = True
        
        # Draw rectangles around detected eyes for visualization
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Update the look-away and look-at times based on eye detection
    current_time = time.time()
    if eyes_detected:
        if not prev_eyes_detected:
            # Transition from looking away to looking at
            look_away_time += current_time - start_state_time
            start_state_time = current_time
    else:
        if prev_eyes_detected:
            # Transition from looking at to looking away
            look_at_time += current_time - start_state_time
            start_state_time = current_time
    
    return frame, eyes_detected, start_state_time

# Main function to capture video from webcam and process each frame
def main():
    global look_away_time, look_at_time, start_time
    cap = cv2.VideoCapture(0)  # Open default webcam
    prev_eyes_detected = False  # Track previous state of eye detection
    start_state_time = start_time  # Track time of current state
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, eyes_detected, start_state_time = detect_eyes(frame, prev_eyes_detected, start_state_time)
        prev_eyes_detected = eyes_detected
        
        # Add the timer and times on the side of the camera feed
        timer_text = f'Time: {int(elapsed_time)}s'
        look_away_text = f'Look Away: {int(look_away_time)}s'
        look_at_text = f'Look At: {int(look_at_time)}s'
        
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, look_away_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, look_at_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the frame with detections and times
        cv2.imshow('Eye Tracker with Timer', frame)
        
        # Check for 'q' key to exit loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Final update for look-at time if still looking at the camera
    if prev_eyes_detected:
        look_at_time += time.time() - start_state_time
    else:
        look_away_time += time.time() - start_state_time
    
    # Release video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print the total times for look-aways and looking at the camera
    print(f"Total time looking away: {int(look_away_time)} seconds")
    print(f"Total time looking at the camera: {int(look_at_time)} seconds")

if __name__ == "__main__":
    main()
