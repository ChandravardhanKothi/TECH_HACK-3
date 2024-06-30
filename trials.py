import urllib.request

# Download the model files (prototxt and caffemodel)
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

urllib.request.urlretrieve(prototxt_url, 'deploy.prototxt')
urllib.request.urlretrieve(caffemodel_url, 'res10_300x300_ssd_iter_140000.caffemodel')
# Load the pre-trained deep learning model for face detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

def detect_face_and_eyes(frame):
    global start_time, countdown_count, warning_active
    height, width = frame.shape[:2]
    
    # Preprocess the frame for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Set the input to the model
    net.setInput(blob)
    
    # Perform inference and get detections
    detections = net.forward()
    
    looking = False
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by confidence threshold
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, w, h) = box.astype("int")
            
            # Ensure the bounding box is within the frame dimensions
            (x, y) = (max(0, x), max(0, y))
            (w, h) = (min(width - x, w), min(height - y, h))
            
            # Extract the face ROI and process it
            face_roi = frame[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Use eye cascade (or further ML methods) for more detailed eye analysis
            eyes = eye_cascade.detectMultiScale(gray_face)
            
            # Check if eyes are detected
            if len(eyes) >= 2:
                looking = True
                start_time = 0  # Reset the start_time if eyes are detected
                warning_active = False  # Reset warning active
                break
            else:
                start_time = time.time() if start_time == 0 else start_time  # Start timer if eyes are not detected
    
    return looking, frame
