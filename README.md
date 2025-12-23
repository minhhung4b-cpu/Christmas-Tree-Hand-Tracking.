# Christmas-Tree-Hand-Tracking.
import cv2
import mediapipe as mp
import numpy as np
import random

# 1. Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 2. Cấu hình hiệu ứng Tuyết rơi
snowflakes = []
for i in range(50):  # Tạo 50 hạt tuyết ban đầu
    snowflakes.append([random.randint(0, 640), random.randint(0, 480), random.randint(2, 5)])

# 3. Đọc file ảnh cây thông
overlay_img = cv2.imread('tree.png', cv2.IMREAD_UNCHANGED)

def apply_color_filter(img, color_bgr):
    """Thay đổi tông màu của cây thông"""
    b, g, r, a = cv2.split(img)
    b = cv2.addWeighted(b, 0.5, b, 0, color_bgr[0])
    g = cv2.addWeighted(g, 0.5, g, 0, color_bgr[1])
    r = cv2.addWeighted(r, 0.5, r, 0, color_bgr[2])
    return cv2.merge((b, g, r, a))

def overlay_transparent(background, overlay, x, y, size=250):
    if overlay is None: return background
    overlay = cv2.resize(overlay, (size, size))
    h, w, _ = overlay.shape
    rows, cols, _ = background.shape
    x, y = x - w // 2, y - h // 2
    if x + w > cols or x < 0 or y + h > rows or y < 0: return background

    overlay_color = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (background[y:y+h, x:x+w, c] * (1 - mask) + 
                                      overlay_color[:, :, c] * mask)
    return background

# 4. Biến điều khiển màu sắc
current_color = (255, 255, 255) # Màu mặc định
colors = [(255, 255, 255), (0, 255, 255), (0, 165, 255), (200, 100, 255)] # Trắng, Vàng, Cam, Hồng
color_idx = 0
cooldown = 0 # Tránh đổi màu quá nhanh

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h_f, w_f, _ = frame.shape
    
    # --- Hiệu ứng Tuyết rơi ---
    for i in range(len(snowflakes)):
        cv2.circle(frame, (snowflakes[i][0], snowflakes[i][1]), snowflakes[i][2], (255, 255, 255), -1)
        snowflakes[i][1] += random.randint(3, 7) # Tuyết rơi xuống
        if snowflakes[i][1] > h_f: # Reset khi chạm đáy
            snowflakes[i][1] = -10
            snowflakes[i][0] = random.randint(0, w_f)

    # --- Nhận diện tay và xử lý cây thông ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Lấy tọa độ ngón cái (4) và ngón giữa (12) để làm hiệu ứng búng tay
            thumb = hand_lms.landmark[4]
            middle = hand_lms.landmark[12]
            
            # Tính khoảng cách giữa ngón cái và ngón giữa
            dist = np.hypot(thumb.x - middle.x, thumb.y - middle.y)
            
            if dist < 0.05 and cooldown == 0: # Nếu chạm 2 ngón vào nhau
                color_idx = (color_idx + 1) % len(colors)
                current_color = colors[color_idx]
                cooldown = 15 # Đợi 15 khung hình mới cho đổi tiếp

            # Vị trí đặt cây thông (điểm mốc số 9)
            cx, cy = int(hand_lms.landmark[9].x * w_f), int(hand_lms.landmark[9].y * h_f)
            
            # Áp dụng màu và vẽ cây
            colored_tree = apply_color_filter(overlay_img, current_color)
            frame = overlay_transparent(frame, colored_tree, cx, cy)

    if cooldown > 0: cooldown -= 1

    cv2.putText(frame, "Cham ngon cai vao ngon giua de doi mau", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Noel Magic", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
