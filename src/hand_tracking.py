import cv2
import mediapipe as mp

from hand_utils import get_finger_states

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------
# Model Path
# ---------------------------------
MODEL_PATH = "assets/models/hand_landmarker.task"

# ---------------------------------
# Create Hand Landmarker
# ---------------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
)

detector = vision.HandLandmarker.create_from_options(options)

# ---------------------------------
# Hand Skeleton Connections
# ---------------------------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ---------------------------------
# Webcam
# ---------------------------------
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    h, w, _ = frame.shape

    cv2.putText(
        frame,
        f"Hands Detected: {len(detection_result.hand_landmarks)}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):

        # print(f"\nHand {hand_index + 1}")

        x_points = []
        y_points = []

        # -------------------------
        # Draw Landmarks
        # -------------------------
        for i, landmark in enumerate(hand_landmarks):

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            x_points.append(x)
            y_points.append(y)

            # print(
            #     f"Landmark {i:2d}: "
            #     f"x={landmark.x:.3f}, "
            #     f"y={landmark.y:.3f}, "
            #     f"z={landmark.z:.3f}"
            # )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0,255,0),
                -1
            )

            cv2.putText(
                frame,
                str(i),
                (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255,255,255),
                1
            )

        # -------------------------
        # Draw Skeleton
        # -------------------------
        for start, end in HAND_CONNECTIONS:

            x1 = int(hand_landmarks[start].x * w)
            y1 = int(hand_landmarks[start].y * h)

            x2 = int(hand_landmarks[end].x * w)
            y2 = int(hand_landmarks[end].y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255,255,255),
                2
            )

        # -------------------------
        # Bounding Box
        # -------------------------
        xmin = min(x_points)
        xmax = max(x_points)
        ymin = min(y_points)
        ymax = max(y_points)

        padding = 20

        cv2.rectangle(
            frame,
            (xmin-padding, ymin-padding),
            (xmax+padding, ymax+padding),
            (255,0,0),
            2
        )

        # -------------------------
        # Center Point
        # -------------------------
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        cv2.circle(
            frame,
            (cx, cy),
            6,
            (0,0,255),
            -1
        )

        # -------------------------
        # Hand Label
        # -------------------------
        handedness = detection_result.handedness[hand_index][0].category_name
        fingers = get_finger_states(hand_landmarks, handedness)

        finger_names = [
            "Thumb",
            "Index",
            "Middle",
            "Ring",
            "Pinky"
        ]

        for i, finger in enumerate(finger_names):

            state = "Open" if fingers[i] else "Closed"

            cv2.putText(
                frame,
                f"{finger}: {state}",
                (xmin, ymax + 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        cv2.putText(
            frame,
            handedness,
            (xmin, ymin-25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,255),
            2
        )

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()