import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Establish a connection to Arduino on COM4
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

# Function to send commands to Arduino
def send_angle(angle):
    arduino.write(f"{angle}\n".encode())  # Send the angle followed by a newline
    print(f"Sent angle: {angle}")
    time.sleep(0.1)  # Short delay to ensure command is processed
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculate the angle at point 'b', with 'a' and 'c' forming the line segments.
    Here 'a' is the stationary point, and 'c' is the moving point.
    """
    ab = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    if norm_ab == 0 or norm_bc == 0:
        return 0.0
    ab_normalized = ab / norm_ab
    bc_normalized = bc / norm_bc
    dot_product = np.dot(ab_normalized, bc_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            img_height, img_width, _ = image.shape
            landmarks = [(lm.x * img_width, lm.y * img_height, lm.z * img_width) for lm in hand_landmarks.landmark]

            # Calculate the angles of the index finger
            mcp_angle = calculate_angle(landmarks[0], landmarks[5], landmarks[6])  # Bottom stationary (0 -> 5 -> 6)
            pip_angle = calculate_angle(landmarks[5], landmarks[6], landmarks[7])  # Bottom stationary (5 -> 6 -> 7)
            dip_angle = calculate_angle(landmarks[6], landmarks[7], landmarks[8])  # Bottom stationary (6 -> 7 -> 8)
            
            angle = pip_angle
            send_angle(angle)  # Send valid angle to Arduino

            # Display the angles of the index finger
            y_offset = 30
            cv2.putText(image, f"Index Finger MCP Angle: {int(mcp_angle)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            cv2.putText(image, f"Index Finger PIP Angle: {int(pip_angle)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            cv2.putText(image, f"Index Finger DIP Angle: {int(dip_angle)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Index Finger Angles', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()


