import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# MediaPipe hand landmark indices (21 points per hand)
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

FORWARD_SPEED = 0.2   # metres per second
TURN_SPEED = 0.5      # radians per second


def finger_extended(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y


def classify_gesture(landmarks):
    """Return one of 'forward', 'back', 'left', 'right' or 'stop'.

    `landmarks` is a list of 21 points, each with `.x` and `.y` in [0, 1].
    x grows to the right, y grows downward.
    """
    index = finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
    middle = finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring = finger_extended(landmarks, RING_TIP, RING_PIP)
    pinky = finger_extended(landmarks, PINKY_TIP, PINKY_PIP)

    if index and not (middle or ring or pinky):
        return "forward"
    if index and middle and ring and pinky:
        return "back"

    fingers_folded = not (index or middle or ring or pinky)
    if fingers_folded:
        dx = landmarks[THUMB_TIP].x - landmarks[INDEX_MCP].x
        if abs(dx) > 0.1:
            return "left" if dx < 0 else "right"

    return "stop"


def gesture_to_twist(gesture):
    """Turn a gesture label into a Twist velocity command."""
    twist = Twist()
    if gesture == "forward":
        twist.linear.x = FORWARD_SPEED
    elif gesture == "back":
        twist.linear.x = -FORWARD_SPEED
    elif gesture == "left":
        twist.angular.z = TURN_SPEED
    elif gesture == "right":
        twist.angular.z = -TURN_SPEED
    return twist


class GestureSteering(Node):
    def __init__(self):
        super().__init__("gesture_steering")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        model_path = os.path.expanduser("~/hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=1)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.capture = cv2.VideoCapture(0)            # 0 = default webcam
        self.timer = self.create_timer(0.1, self.process_frame)   # 10 Hz

    def process_frame(self):
        ok, frame = self.capture.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)                    # mirror, so it feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        gesture = "stop"
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            gesture = classify_gesture(landmarks)
            self.draw_landmarks(frame, landmarks)

        self.publisher.publish(gesture_to_twist(gesture))

        cv2.putText(frame, gesture, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Gesture steering", frame)
        cv2.waitKey(1)

    def draw_landmarks(self, frame, landmarks):
        h, w = frame.shape[:2]
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)


def main():
    rclpy.init()
    node = GestureSteering()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publisher.publish(Twist())     # stop the robot on exit
        node.capture.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
