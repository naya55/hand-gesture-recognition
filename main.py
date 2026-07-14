import cv2
from mediapipe.python.solutions import hands
from mediapipe.python.solutions import drawing_utils

mp_hands = hands
mp_draw = drawing_utils

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
YELLOW = (0,255,255)
PURPLE = (255,0,255)
WHITE = (255,255,255)

def draw_text(frame, text, y, color):

    cv2.putText(
        frame,
        text,
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

def blur_frame(frame):

    return cv2.GaussianBlur(
        frame,
        (101,101),
        0
    )

def get_finger_status(landmarks):

    thumb_up = landmarks[4].x < landmarks[3].x

    index_up = landmarks[8].y < landmarks[6].y

    middle_up = landmarks[12].y < landmarks[10].y

    ring_up = landmarks[16].y < landmarks[14].y

    pinky_up = landmarks[20].y < landmarks[18].y

    return (
        thumb_up,
        index_up,
        middle_up,
        ring_up,
        pinky_up
    )

def detect_gesture(landmarks):

    thumb_up, index_up, middle_up, ring_up, pinky_up = get_finger_status(landmarks)

    if index_up and middle_up and (not ring_up) and (not pinky_up):
        return "PEACE DETECTED"

    elif thumb_up and index_up and (not middle_up) and (not ring_up) and pinky_up:
        return "ILY"

    elif thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "HALO"

    elif (not thumb_up) and index_up and (not middle_up) and (not ring_up) and (not pinky_up):
        return "AKU"

    elif thumb_up and (not index_up) and (not middle_up) and (not ring_up) and pinky_up:
        return "NAYA"

    return None

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand_detector.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            gesture = detect_gesture(landmarks)

            if gesture == "PEACE DETECTED":

                frame = blur_frame(frame)
                draw_text(frame, "PEACE DETECTED", 50, GREEN)

            elif gesture == "ILY":

                draw_text(frame, "I LOVE YOU <3", 50, RED)

            elif gesture == "HALO":

                draw_text(frame, "HALO", 50, YELLOW)

            elif gesture == "AKU":

                draw_text(frame, "AKU", 50, BLUE)

            elif gesture == "NAYA":

                draw_text(frame, "NAYA", 50, PURPLE)
    
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()