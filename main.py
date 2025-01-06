# Import necessary libraries
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame  # For playing and stopping audio
import os

# Initialize pygame mixer
pygame.mixer.init()

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Paths to the shape predictor and alert sound
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ALERT_SOUND_PATH = os.path.join("assets", "alert_sound.mp3")

# Load face and eye detection models
face_detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Drowsiness detection parameters
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 48
COUNTER = 0
is_drowsy = False  # Flag to track drowsy state

# Load sound only once
alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get coordinates for eyes
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw eye landmarks (Optional: for visualization)
        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Check if EAR is below threshold (indicating drowsiness)
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EAR_CONSEC_FRAMES:
                is_drowsy = True  # Set the flag to True to indicate drowsiness
                # DROWSY message with background and better font
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 255), -1)  # Red background
                cv2.putText(frame, "DROWSY!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # White text
                if not pygame.mixer.get_busy():  # Check if sound is not already playing
                    alert_sound.play(-1)  # Play sound in a loop
        else:
            COUNTER = 0
            is_drowsy = False  # Set the flag to False to indicate awake state
            # AWAKE message with background and better font
            cv2.rectangle(frame, (10, 10), (400, 80), (0, 255, 0), -1)  # Green background
            cv2.putText(frame, "AWAKE", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # White text
            if pygame.mixer.get_busy():  # If sound is playing, stop it
                alert_sound.stop()

    # Display the frame
    cv2.imshow("Drowsiness Alert System", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # Quit pygame mixer
