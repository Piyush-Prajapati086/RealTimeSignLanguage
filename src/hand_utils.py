# Finger tip landmark IDs
TIP_IDS = [4, 8, 12, 16, 20]


def get_finger_states(hand_landmarks, handedness):
    """
    Returns:
    [Thumb, Index, Middle, Ring, Pinky]

    1 = Open
    0 = Closed
    """

    fingers = []

    # -----------------------------
    # Thumb
    # -----------------------------
    if handedness == "Right":
        fingers.append(
            1 if hand_landmarks[4].x > hand_landmarks[3].x else 0
        )
    else:
        fingers.append(
            1 if hand_landmarks[4].x < hand_landmarks[3].x else 0
    )

    # -----------------------------
    # Index, Middle, Ring, Pinky
    # -----------------------------
    for tip in TIP_IDS[1:]:

        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers