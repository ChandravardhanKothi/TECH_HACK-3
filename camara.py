import cv2

# Load the HandDetector model
hand_detector = cv2.dnn.readNetFromTensorflow('hand_detector.pb')

# Open the default camera
cap = cv2.VideoCapture(0)

# Loop until the user presses the Esc key
while True:

    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to a smaller size for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the hand in the frame
    detections = hand_detector.detectMultiScale(gray, 1.1, 10)

    # Loop over the detections
    for (x, y, w, h) in detections:

        # Draw a rectangle around the hand
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Count the number of fingers
        fingers = count_fingers(frame, x, y, w, h)

        # Draw the number of fingers on the frame
        cv2.putText(frame, str(fingers), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
