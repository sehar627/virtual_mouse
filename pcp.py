import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Store all drawn points
draw_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip coordinates
            x = int(hand_landmarks.landmark[8].x * width)
            y = int(hand_landmarks.landmark[8].y * height)

            # Add to draw points list
            draw_points.append((x, y))

    # Draw all points as a continuous line
    for i in range(1, len(draw_points)):
        cv2.circle(frame, draw_points[i-1], draw_points[i], (0, 255, 0), 5)

    cv2.imshow("Air Drawing (No Canvas)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear all drawn points
        draw_points = []

cap.release()
cv2.destroyAllWindows()
