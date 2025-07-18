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

try:
    while True:
        angle = input("Enter an angle (0-180) or 'q' to quit: ").strip()
        if angle.lower() == 'q':
            print("Exiting...")
            break
        if angle.isdigit() and 0 <= int(angle) <= 180:
            send_angle(angle)  # Send valid angle to Arduino
        else:
            print("Invalid input. Please enter a number between 0 and 180.")
finally:
    arduino.close()  # Close the connection when done
