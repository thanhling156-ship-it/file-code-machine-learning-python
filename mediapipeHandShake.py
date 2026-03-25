import cv2
import mediapipe as mp
import numpy as np
import os
import pyautogui
import time # Thư viện then chốt cho phần Time

# --- SETUP HỆ THỐNG (Giữ nguyên từ bản trước) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'hand_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

# --- BIẾN THỜI GIAN (Dành cho Architect) ---
last_action_time = 0
action_delay = 0.2  # Tốc độ tăng/giảm (0.2 giây một nấc)
last_count = -1

def count_fingers_fixed(landmarks, handedness):
    hand_label = handedness[0].category_name
    tips = [8, 12, 16, 20]; knuckles = [6, 10, 14, 18]
    raised = [landmarks[t].y < landmarks[k].y for t, k in zip(tips, knuckles)]
    
    # Logic ngón cái đã sửa từ ảnh trước
    thumb_tip = landmarks[4]; thumb_ip = landmarks[3]
    is_thumb = (thumb_tip.x < thumb_ip.x - 0.01) if hand_label == "Right" else (thumb_tip.x > thumb_ip.x + 0.01)
    raised.insert(0, is_thumb)
    return sum(raised)

# --- CHƯƠNG TRÌNH CHÍNH ---
detector = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        count = count_fingers_fixed(result.hand_landmarks[0], result.handedness[0])
        current_time = time.time()

        # --- LOGIC ĐIỀU KHIỂN THEO THỜI GIAN ---
        # 1. Tăng/Giảm Volume (Cần lặp lại khi GIỮ ngón tay)
        if count in [1, 2]:
            if current_time - last_action_time > action_delay:
                if count == 1:
                    pyautogui.press('volumeup')
                    status = "Increasing Volume..."
                else:
                    pyautogui.press('volumedown')
                    status = "Decreasing Volume..."
                last_action_time = current_time # Reset đồng hồ
        
        # 2. Lệnh đặc biệt (Chỉ thực hiện 1 LẦN khi vừa giơ tay)
        elif count == 5 and count != last_count:
            pyautogui.hotkey('ctrl', 'shift', 'esc')
            status = "Task Manager Opened"
            last_count = count
        
        # 3. Reset trạng thái khi nắm tay (0 ngón)
        elif count == 0:
            status = "System Ready"
            last_count = 0
        else:
            status = f"Holding {count} fingers"

        cv2.putText(frame, status, (10, 80), 2, 1, (0, 255, 0), 2)
    else:
        last_count = -1 # Reset khi không thấy tay

    cv2.imshow('Temporal AI Controller', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()