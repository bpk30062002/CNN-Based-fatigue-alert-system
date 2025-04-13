import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer

# Initialize Pygame mixer for audio
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')

# Load Haar Cascade files for face and eye detection
face_detection = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_detection = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_detection = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load the pre-trained model
model = load_model('models/custmodel.h5')

# Set up video capture
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise IOError("Cannot open webcam")

# Set up font and variables
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
counter = 0
inactive_time = 0
thickness = 2

while True:
    ret, frame = capture.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face_detection.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eyes = left_eye_detection.detectMultiScale(gray)
    right_eyes = right_eye_detection.detectMultiScale(gray)

    # Draw rectangles for faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    # Initialize eye state
    right_eye_state = None
    left_eye_state = None

    # Process right eye
    for (x, y, w, h) in right_eyes:
        right_eye = frame[y:y+h, x:x+w]
        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        right_eye = cv2.resize(right_eye, (24, 24)) / 255.0
        right_eye = right_eye.reshape(1, 24, 24, 1)
        prediction = model.predict(right_eye)
        right_eye_state = np.argmax(prediction, axis=1)[0]
        break

    # Process left eye
    for (x, y, w, h) in left_eyes:
        left_eye = frame[y:y+h, x:x+w]
        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        left_eye = cv2.resize(left_eye, (24, 24)) / 255.0
        left_eye = left_eye.reshape(1, 24, 24, 1)
        prediction = model.predict(left_eye)
        left_eye_state = np.argmax(prediction, axis=1)[0]
        break

    # Determine state and time
    if right_eye_state == 0 and left_eye_state == 0:
        inactive_time += 1
        cv2.putText(frame, "Inactive", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        inactive_time -= 1
        cv2.putText(frame, "Active", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    inactive_time = max(inactive_time, 0)
    cv2.putText(frame, f'Wake up Time !!: {inactive_time}', (300, height-20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    if inactive_time > 10:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            alarm_sound.play()
        except:
            pass

        # Increase or decrease rectangle thickness
        if thickness < 16:
            thickness += 2
        else:
            thickness -= 2
            if thickness < 2:
                thickness = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()