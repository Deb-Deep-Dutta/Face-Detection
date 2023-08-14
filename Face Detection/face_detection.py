# Face dewtection using Haar Cascade

import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image or capture from webcam
# If you're using an image, replace '0' with the image filename
# If you're using a webcam, replace '0' with the appropriate camera index
# For my machine, I was using a webcam and '0' worked.
capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
     # Apply histogram equalization to enhance contrast in low-light conditions
    gray = cv2.equalizeHist(gray)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
capture.release()
cv2.destroyAllWindows()
