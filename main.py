"""Show a reaction photo when a webcam gesture holds for seven frames."""

import argparse
from collections import deque
from contextlib import ExitStack
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np


STABLE_FRAMES = 7
POSE_VISIBILITY = 0.60
FRAME_EDGE = 0.015
FOREHEAD_EYE_SPAN = 0.70
OVERHEAD_CLEARANCE = 0.04
WRIST_ABOVE_ELBOW = 0.05
SHRUG_ELBOW_MAX = 145
SHRUG_WRIST_OUTSIDE = 0.10
SHRUG_WRIST_BELOW_SHOULDER = 1.15
SHRUG_WRIST_BELOW_ELBOW = 0.25
SHRUG_OPEN_REACH = 0.88
HAND_WRIST_MATCH = 0.40
PALM_FACING_MIN = 0.40
OPEN_PIP = 155
OPEN_DIP = 150
CURLED_PIP = 145
CURLED_REACH = 0.80
PEACE_V_ANGLE = 12
PEACE_TIP_GAP = 0.36
BECKON_INDEX_PIP = (65, 162)
BECKON_INDEX_REACH = (0.37, 0.98)
BECKON_INDEX_GAP = 0.45
BECKON_NEAREST_TIP_GAP = 0.12
BECKON_OTHER_PIP = 156
BECKON_OTHER_REACH = 0.98
BECKON_MIN_OTHER_OPENNESS = 0.55

WAITING = "Waiting for pose"
ASSET_NAMES = {
    "Shrug": "IMG_5342.jpg",
    "Peace sign": "IMG_5343.jpg",
    "Hands up": "IMG_5344.JPG",
    "Beckon": "IMG_5345.JPG",
}
ASSET_DIR = Path(__file__).resolve().parent / "assets" / "new"
WEBCAM_WINDOW = "MemeCV — Webcam"
REACTION_WINDOW = "MemeCV — Reaction"
PINK = (180, 105, 255)
POSE = mp.solutions.pose.PoseLandmark
ARM_NAMES = [
    (POSE.LEFT_SHOULDER, POSE.LEFT_ELBOW, POSE.LEFT_WRIST),
    (POSE.RIGHT_SHOULDER, POSE.RIGHT_ELBOW, POSE.RIGHT_WRIST),
]
FACE_POINTS = (POSE.LEFT_EYE_OUTER, POSE.RIGHT_EYE_OUTER, POSE.NOSE)
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def angle(first, vertex, last):
    left, right = first - vertex, last - vertex
    length = np.linalg.norm(left) * np.linalg.norm(right)
    if length < 1e-8:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(left, right) / length, -1, 1)))


class Hand:
    def __init__(self, landmarks, aspect):
        # MediaPipe hand depth uses image-width units, as does x.
        self.points = np.array([(p.x * aspect, p.y, p.z * aspect) for p in landmarks])
        self.width = max(np.linalg.norm(self.points[5] - self.points[17]), 1e-8)
        self.pip = []
        self.dip = []
        self.reach = []
        for mcp, pip, dip, tip in self.points[list(FINGERS)]:
            self.pip.append(angle(mcp, pip, dip))
            self.dip.append(angle(pip, dip, tip))
            chain = np.linalg.norm(mcp - pip) + np.linalg.norm(pip - dip) + np.linalg.norm(dip - tip)
            self.reach.append(np.linalg.norm(tip - mcp) / max(chain, 1e-8))

    def extended(self, finger):
        return self.pip[finger] >= OPEN_PIP and self.dip[finger] >= OPEN_DIP

    def curled(self, finger):
        return self.pip[finger] <= CURLED_PIP and self.reach[finger] <= CURLED_REACH

    def palm_angled(self):
        # An unsigned normal tolerates both hands and mirror reversal.
        wrist = self.points[0]
        normal = np.cross(self.points[5] - wrist, self.points[17] - wrist)
        length = np.linalg.norm(normal)
        return length > 1e-8 and max(abs(normal[1]), abs(normal[2])) / length >= PALM_FACING_MIN


def peace(hand):
    p = hand.points
    separation = np.linalg.norm(p[8] - p[12]) / hand.width
    v_angle = angle(p[8], p[5], p[5] + (p[12] - p[9]))
    return (hand.extended(0) and hand.extended(1) and hand.curled(2)
            and hand.curled(3) and separation >= PEACE_TIP_GAP and v_angle >= PEACE_V_ANGLE)


def beckon(hand):
    p = hand.points
    other_tips = p[[12, 16, 20]]
    group_gap = np.linalg.norm(p[8] - other_tips.mean(axis=0)) / hand.width
    nearest_gap = np.linalg.norm(p[8] - other_tips, axis=1).min() / hand.width
    thumb_up = (angle(p[2], p[3], p[4]) >= OPEN_DIP
                and p[4, 1] < min(p[3, 1], p[5, 1]) - 0.20 * hand.width
                and all(hand.curled(i) for i in range(4)))
    loosely_curled = sum(hand.pip[i] <= BECKON_OTHER_PIP
                         and hand.reach[i] < BECKON_OTHER_REACH for i in (1, 2, 3))
    return (hand.palm_angled() and not thumb_up
            and BECKON_INDEX_PIP[0] <= hand.pip[0] <= BECKON_INDEX_PIP[1]
            and BECKON_INDEX_REACH[0] <= hand.reach[0] <= BECKON_INDEX_REACH[1]
            and group_gap >= BECKON_INDEX_GAP and nearest_gap >= BECKON_NEAREST_TIP_GAP
            and loosely_curled >= 2
            and max(hand.reach[1:]) >= BECKON_MIN_OTHER_OPENNESS)


def visible_points(landmarks, indices, aspect):
    result = []
    for i in indices:
        p = landmarks[i]
        if (p.visibility < POSE_VISIBILITY
                or not FRAME_EDGE <= p.x <= 1 - FRAME_EDGE
                or not FRAME_EDGE <= p.y <= 1 - FRAME_EDGE):
            return None
        result.append(np.array((p.x * aspect, p.y)))
    return result


def hands_up(landmarks, arms, aspect):
    if arms is None:
        return False
    face = visible_points(landmarks, FACE_POINTS, aspect)
    if face is None:
        return False
    left_eye, right_eye, nose = face
    eye_span = np.linalg.norm(left_eye - right_eye)
    top_of_face = min(left_eye[1], right_eye[1], nose[1]) - FOREHEAD_EYE_SPAN * eye_span
    shoulder_width = np.linalg.norm(arms[0][0] - arms[1][0])
    return top_of_face >= FRAME_EDGE and shoulder_width > 1e-8 and all(
        wrist[1] <= top_of_face - OVERHEAD_CLEARANCE * shoulder_width
        and wrist[1] <= elbow[1] - WRIST_ABOVE_ELBOW * shoulder_width
        for _, elbow, wrist in arms
    )


def shrug(arms, hands):
    if arms is None or len(hands) != 2:
        return False
    center_x = (arms[0][0][0] + arms[1][0][0]) / 2
    width = abs(arms[0][0][0] - arms[1][0][0])
    if width < 1e-8:
        return False
    for shoulder, elbow, wrist in arms:
        side = np.sign(shoulder[0] - center_x)
        if (side * (wrist[0] - shoulder[0]) < SHRUG_WRIST_OUTSIDE * width
                or wrist[1] > elbow[1] + SHRUG_WRIST_BELOW_ELBOW * width
                or wrist[1] > shoulder[1] + SHRUG_WRIST_BELOW_SHOULDER * width
                or angle(shoulder, elbow, wrist) > SHRUG_ELBOW_MAX):
            return False
    # Test both pairings so MediaPipe's hand list order is irrelevant.
    for ordered_hands in (hands, hands[::-1]):
        if all(np.linalg.norm(hand.points[0, :2] - arm[2]) <= HAND_WRIST_MATCH * width
               and np.sign(hand.points[0, 0] - center_x) == np.sign(arm[0][0] - center_x)
               and hand.palm_angled()
               and all(reach >= SHRUG_OPEN_REACH for reach in hand.reach)
               for arm, hand in zip(arms, ordered_hands)):
            return True
    return False


def candidate_for(pose, hands, aspect):
    arms = None
    if pose is not None:
        points = [visible_points(pose, indices, aspect) for indices in ARM_NAMES]
        if all(arm is not None for arm in points):
            arms = points
    # The requested timeout priority slot corresponds to Hands up.
    if hands_up(pose, arms, aspect):
        return "Hands up"
    if shrug(arms, hands):
        return "Shrug"
    if any(peace(hand) for hand in hands):
        return "Peace sign"
    if sum(beckon(hand) for hand in hands) == 1:
        return "Beckon"
    return WAITING


def confirm(history, candidate, previous):
    history.append(candidate)
    if len(history) == STABLE_FRAMES and all(value == candidate for value in history):
        return candidate
    return previous


def load_reactions():
    reactions = {}
    for gesture, filename in ASSET_NAMES.items():
        path = ASSET_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing reaction photo: {path}")
        photo = cv2.imread(str(path))
        if photo is None:
            raise ValueError(f"Unreadable reaction photo: {path}")
        height, width = photo.shape[:2]
        scale = min(1, 840 / width, 640 / height)
        if scale < 1:
            photo = cv2.resize(photo, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_AREA)
        reactions[gesture] = photo
    card = np.full((360, 640, 3), (36, 28, 32), dtype=np.uint8)
    label_width = cv2.getTextSize(WAITING, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0]
    cv2.putText(card, WAITING, ((640 - label_width) // 2, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 245, 250), 2, cv2.LINE_AA)
    reactions[WAITING] = card
    return reactions


def open_camera(index):
    backends = (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY) if sys.platform == "darwin" else (cv2.CAP_ANY,)
    for backend in backends:
        camera = cv2.VideoCapture()
        try:
            if camera.open(index, backend):
                for _ in range(8):
                    ok, frame = camera.read()
                    if ok and frame is not None and frame.size:
                        return camera, frame
        except cv2.error:
            pass
        camera.release()
    raise RuntimeError("Could not open the camera or read a frame.")


def draw_points(frame, landmarks, color):
    height, width = frame.shape[:2]
    for p in landmarks:
        if 0 <= p.x < 1 and 0 <= p.y < 1:
            cv2.circle(frame, (int(p.x * width), int(p.y * height)), 2, color, -1, cv2.LINE_AA)


def run(index, show_face):
    reactions = load_reactions()
    camera = None
    history = deque(maxlen=STABLE_FRAMES)
    confirmed = WAITING
    try:
        camera, frame = open_camera(index)
        with ExitStack() as stack:
            pose_model = stack.enter_context(mp.solutions.pose.Pose(model_complexity=1))
            hand_model = stack.enter_context(mp.solutions.hands.Hands(max_num_hands=2))
            face_model = stack.enter_context(mp.solutions.face_mesh.FaceMesh()) if show_face else None
            cv2.namedWindow(WEBCAM_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(REACTION_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(REACTION_WINDOW, reactions[confirmed])
            cv2.waitKey(1)
            while True:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                pose = pose_model.process(rgb).pose_landmarks
                hand_sets = hand_model.process(rgb).multi_hand_landmarks or []
                aspect = frame.shape[1] / frame.shape[0]
                hands = [Hand(result.landmark, aspect) for result in hand_sets
                         if all(FRAME_EDGE <= p.x <= 1 - FRAME_EDGE
                                and FRAME_EDGE <= p.y <= 1 - FRAME_EDGE for p in result.landmark)]
                candidate = candidate_for(pose.landmark if pose else None, hands, aspect)
                confirmed = confirm(history, candidate, confirmed)
                if face_model:
                    for result in face_model.process(rgb).multi_face_landmarks or []:
                        draw_points(frame, result.landmark, (255, 255, 255))
                for result in hand_sets:
                    draw_points(frame, result.landmark, PINK)
                for row, label in enumerate((f"Candidate: {candidate}", f"Confirmed: {confirmed}")):
                    position = (12, 26 + 25 * row)
                    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WEBCAM_WINDOW, frame)
                cv2.imshow(REACTION_WINDOW, reactions[confirmed])
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                ok, frame = camera.read()
                if not ok or frame is None or not frame.size:
                    raise RuntimeError("The camera stopped producing frames.")
    finally:
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--face-landmarks", action="store_true", help="draw white face landmarks")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be zero or greater")
    try:
        run(args.camera, args.face_landmarks)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        print(f"MemeCV: {error}", file=sys.stderr)
        if isinstance(error, RuntimeError):
            print("On macOS, enable camera access for Terminal or your launching IDE under "
                  "System Settings → Privacy & Security → Camera. Close other apps using "
                  "the camera, then retry or try another index with --camera 1.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
