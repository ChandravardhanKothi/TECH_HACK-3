import cv2
import time
import threading
import random
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open a connection to the default camera (usually the first camera)
vid = cv2.VideoCapture(0)

# Global variables
exit_camera = False
start_time = 0
countdown_count = 0
warning_active = False
question_active = False
question = ""
options = []
correct_option = -1

# Function to detect face and eyes, and determine if person is looking at the camera
def detect_face_and_eyes(frame):
    global start_time, countdown_count, warning_active, question_active
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    looking = False
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Check if eyes are detected
            if len(eyes) >= 2:
                looking = True
                start_time = 0  # Reset the start_time if eyes are detected
                if not question_active:
                    warning_active = False  # Reset warning active
                    question_active = False  # Reset question active
                break
            else:
                start_time = time.time() if start_time == 0 else start_time  # Start timer if eyes are not detected
    else:
        start_time = time.time() if start_time == 0 else start_time  # Start timer if no faces detected
    
    return looking, frame

# Function to handle the countdown timer
def handle_timer():
    global exit_camera, start_time, countdown_count, warning_active, question_active, question, options, correct_option
    while True:
        if start_time > 0 and not warning_active:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 5:
                countdown_count += 1
                if countdown_count < 3:
                    warning_active = True
                    question_active = True
                    start_time = 0  # Reset the timer
                    # Generate a math question
                    num1 = random.randint(1, 10)
                    num2 = random.randint(1, 10)
                    if random.choice([True, False]):
                        question = f"{num1} + {num2} = ?"
                        correct_option = num1 + num2
                    else:
                        question = f"{num1} * {num2} = ?"
                        correct_option = num1 * num2
                    options = [correct_option, correct_option + 1, correct_option - 1, correct_option + 2]
                    random.shuffle(options)
                else:
                    exit_camera = True
                    break
        time.sleep(1)

# Function to detect the number of fingers
def count_fingers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0

    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour)
    defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))
    
    if defects is None:
        return 0
    
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        a = np.linalg.norm(np.array(start) - np.array(end))
        b = np.linalg.norm(np.array(start) - np.array(far))
        c = np.linalg.norm(np.array(end) - np.array(far))
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        
        if angle <= 90:
            finger_count += 1
    
    return finger_count

# Start the timer in a separate thread
timer_thread = threading.Thread(target=handle_timer)
timer_thread.start()

while True:
    ret, frame = vid.read()
    if not ret:
        break  # Break the loop if there is an issue with the camera feed
    
    # Detect face and eyes, and determine if person is looking at the camera
    looking, processed_frame = detect_face_and_eyes(frame)
    
    # Display the timer countdown on the frame
    if start_time > 0:
        text = f"Time left: {max(0, 5 - int(time.time() - start_time))} seconds"
        cv2.putText(processed_frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # Increased font size and thickness
    
    # Display question if necessary
    if question_active:
        red_scale_frame = frame.copy()
        red_scale_frame[:, :, 1] = 0  # Zero out the green channel
        red_scale_frame[:, :, 0] = 0  # Zero out the blue channel
        processed_frame = red_scale_frame
        cv2.putText(processed_frame, "ANSWER THE QUESTION", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(processed_frame, question, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        for i, option in enumerate(options):
            cv2.putText(processed_frame, f"{option}", (50, 250 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Detect the number of fingers
        finger_count = count_fingers(frame)
        if finger_count > 0:
            if options[finger_count - 1] == correct_option:
                question_active = False  # Reset question active
                countdown_count = 0  # Reset the countdown count
                start_time = 0  # Reset the timer
            else:
                exit_camera = True
                break
    
    # Display the resulting frame
    cv2.imshow('Frame', processed_frame)
    
    # Break the loop when 'q' key is pressed or if exit_camera is True
    if cv2.waitKey(1) & 0xFF == ord('q') or exit_camera:
        break

# Release the video capture object and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()

# Show dismissal message if necessary
if exit_camera:
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Dismissed", "You are dismissed.")
    root.destroy()
