import cv2
import mediapipe as mp

def run_hand_detection():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera and MediaPipe Hand Detection active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        finger_tip_coords = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = frame.shape

                index_tip_x = int(index_finger_tip.x * w)
                index_tip_y = int(index_finger_tip.y * h)
                finger_tip_coords.append((index_tip_x, index_tip_y))

                cv2.circle(frame, (index_tip_x, index_tip_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, f'Index: ({index_tip_x}, {index_tip_y})',
                            (index_tip_x + 15, index_tip_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Hand Detection Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Hand detection test finished.")

if __name__ == "__main__":
    run_hand_detection()