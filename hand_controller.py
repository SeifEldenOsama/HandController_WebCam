import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from pynput.keyboard import Controller, Key


CAM_INDEX = 0            
SMOOTH_FRAMES = 6        
SWIPE_THRESHOLD = 80     
COOLDOWN_TIME = 0.8      
DRAW_LANDMARKS = True


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

keyboard = Controller()

def detect_swipe(movement_history):
    if len(movement_history) < 2:
        return None

    x_start, y_start = movement_history[0]
    x_end, y_end = movement_history[-1]

    dx = x_end - x_start
    dy = y_end - y_start


    if abs(dx) > SWIPE_THRESHOLD and abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"


    if abs(dy) > SWIPE_THRESHOLD and abs(dy) > abs(dx):
        return "down" if dy > 0 else "up"

    return None

def perform_action(action):
    if action == "left":
        keyboard.press(Key.left)
        keyboard.release(Key.left)
        print("⬅ Swipe LEFT detected")
    elif action == "right":
        keyboard.press(Key.right)
        keyboard.release(Key.right)
        print("➡ Swipe RIGHT detected")
    elif action == "up":
        keyboard.press(Key.up)
        keyboard.release(Key.up)
        print("⬆ Swipe UP detected")
    elif action == "down":
        keyboard.press(Key.down)
        keyboard.release(Key.down)
        print("⬇ Swipe DOWN detected")

def main():
    global DRAW_LANDMARKS
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Change CAM_INDEX if needed.")

    movement_history = deque(maxlen=SMOOTH_FRAMES)
    last_action_time = time.time()

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                   
                    h, w, _ = frame.shape
                    cx = int(hand_landmarks.landmark[9].x * w)
                    cy = int(hand_landmarks.landmark[9].y * h)

                    movement_history.append((cx, cy))

                  
                    if DRAW_LANDMARKS:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

              
                action = detect_swipe(movement_history)
                if action and time.time() - last_action_time > COOLDOWN_TIME:
                    perform_action(action)
                    last_action_time = time.time()
                    movement_history.clear()

          
            cv2.putText(frame, "Swipe: LEFT / RIGHT / UP / DOWN", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press Z to toggle landmarks | ESC/Q to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("Hand Swipe Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): 
                break
            elif key == ord('z'):
                DRAW_LANDMARKS = not DRAW_LANDMARKS

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
