import cv2
import mediapipe as mp
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Finger tip landmarks (index: 8, middle: 12, ring: 16, pinky: 20, thumb: 4)
FINGER_TIPS = [4, 8, 12, 16, 20]

# Function to check if only the middle finger is up
def is_middle_finger_up(hand_landmarks):
    # Get y-coordinates of finger tips and pip joints
    tips = [hand_landmarks.landmark[i].y for i in FINGER_TIPS]
    pips = [hand_landmarks.landmark[i-2].y for i in FINGER_TIPS]
    # Thumb: tip should be to the right of pip for right hand (x axis)
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_ip_x = hand_landmarks.landmark[3].x
    # Check fingers
    fingers = []
    # Thumb
    fingers.append(thumb_tip_x > thumb_ip_x)
    # Other fingers: tip is above pip (y is less)
    for i in range(1, 5):
        fingers.append(tips[i] < pips[i])
    # Only middle finger up
    return fingers == [False, False, True, False, False]

# Start webcam
cap = cv2.VideoCapture(0)
shut_down_triggered = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip for selfie view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_middle_finger_up(hand_landmarks):
                cv2.putText(frame, 'Middle Finger Detected! Shutting down...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Shut Down on Middle Finger', frame)
                shut_down_triggered = True
                break
    cv2.imshow('Shut Down on Middle Finger', frame)
    if shut_down_triggered:
        time.sleep(2)  # Show message for 2 seconds
        os.system('shutdown /s /t 1')
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




