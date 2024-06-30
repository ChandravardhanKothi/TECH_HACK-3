import cv2
import time
import threading
import tkinter as tk
from tkinter import messagebox
import random

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open a connection to the default camera (usually the first camera)
vid = cv2.VideoCapture(0)

# Global variables
show_question_message = False
exit_camera = False
start_time = 0
countdown_count = 0
question_active = False
random_sum = 0

# Function to detect face and eyes, and determine if person is looking at the camera
def detect_face_and_eyes(frame):
    global start_time, countdown_count, question_active
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
                question_active = False  # Reset question active
                break
            else:
                start_time = time.time() if start_time == 0 else start_time  # Start timer if eyes are not detected
    else:
        start_time = time.time() if start_time == 0 else start_time  # Start timer if no faces detected
    
    return looking, frame

# Function to handle the countdown timer
def handle_timer():
    global show_question_message, exit_camera, start_time, countdown_count, question_active, random_sum
    while True:
        if start_time > 0 and not question_active:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 5:
                countdown_count += 1
                if countdown_count < 3:
                    question_active = True
                    random_sum = random.randint(1, 10) + random.randint(1, 10)
                else:
                    show_question_message = True
                    exit_camera = True
                    break
                start_time = 0  # Reset the timer
        time.sleep(1)

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
    
    # Display the random sum if necessary
    if question_active:
        red_scale_frame = frame.copy()
        red_scale_frame[:, :, 1] = 0  # Zero out the green channel
        red_scale_frame[:, :, 0] = 0  # Zero out the blue channel
        processed_frame = red_scale_frame
        cv2.putText(processed_frame, f"Question: {random_sum} (Answer with keys)", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Display the resulting frame
    cv2.imshow('Frame', processed_frame)
    
    # Check for user input to answer the sum
    if question_active and cv2.waitKey(1) & 0xFF in [ord(str(i)) for i in range(10)]:
        question_active = False
        countdown_count = 0  # Reset countdown count
    
    # Break the loop when 'q' key is pressed or if exit_camera is True
    if cv2.waitKey(1) & 0xFF == ord('q') or exit_camera:
        break

# Release the video capture object and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()

# Show dismissal message if necessary
if show_question_message:
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Dismissed", "You are dismissed.")
    root.destroy()
