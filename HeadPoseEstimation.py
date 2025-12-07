import cv2
import mediapipe as mp
import numpy as np
import time
from pynput.keyboard import Controller, Key
    

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
keyboard = Controller()

left_right_threshold = 40   
pitch_forward_threshold = 15   
pitch_brake_threshold = -25    
offtrack_yaw_limit = 50
offtrack_pitch_limit = 40

prev_steering = None
prev_throttle = None
prev_brake = None


while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    steering_key = None
    throttle_key = None
    brake_key = None
    off_track = False
    text = "Forward"

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            left_eye_corner = (lm[33].x * img_w, lm[33].y * img_h)
            right_eye_corner = (lm[263].x * img_w, lm[263].y * img_h)

            eye_mid_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
            face_center_x = img_w / 2
            eye_offset = eye_mid_x - face_center_x

            max_yaw_angle = 35
            yaw_deg = (eye_offset / (img_w / 2)) * max_yaw_angle

            image_points = np.array([
                (lm[1].x   * img_w, lm[1].y   * img_h),
                (lm[199].x * img_w, lm[199].y * img_h),
                (lm[33].x  * img_w, lm[33].y  * img_h),
                (lm[263].x * img_w, lm[263].y * img_h),
                (lm[61].x  * img_w, lm[61].y  * img_h),
                (lm[291].x * img_w, lm[291].y * img_h)
            ], dtype=np.float64)

            object_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ], dtype=np.float64)

            focal_length = img_w
            cam_matrix = np.array([
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                object_points, image_points, cam_matrix, dist_matrix,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch_deg = angles[0] * 180

            if eye_offset < -left_right_threshold:
                steering_key = Key.left
                text = "Left"
            elif eye_offset > left_right_threshold:
                steering_key = Key.right
                text = "Right"
            else:
                steering_key = None

            if pitch_deg < pitch_brake_threshold:
                throttle_key = None
                brake_key = Key.down
                if steering_key is None:
                    text = "Brake"

            elif pitch_deg > pitch_forward_threshold:
                throttle_key = Key.up
                brake_key = None
                if steering_key is None:
                    text = "Forward (Boost)"

            else:
                throttle_key = None
                brake_key = None
                if steering_key is None:
                    text = "Forward"

            if abs(yaw_deg) > offtrack_yaw_limit or abs(pitch_deg) > offtrack_pitch_limit:
                off_track = True

            cv2.putText(image, text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

            cv2.putText(image, f"Yaw: {yaw_deg:.1f}°", (20,140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)

            cv2.putText(image, f"Pitch: {pitch_deg:.1f}°", (20,180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,200,0), 2)

            cv2.putText(image, f"Off-track: {off_track}", (20,220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255) if off_track else (0,255,0), 2)

            fps = 1 / (time.time() - start)
            cv2.putText(image, f'FPS: {int(fps)}', (20,450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    if steering_key != prev_steering:
        if prev_steering == Key.left:
            keyboard.release(Key.left)
        elif prev_steering == Key.right:
            keyboard.release(Key.right)

        if steering_key == Key.left:
            keyboard.press(Key.left)
        elif steering_key == Key.right:
            keyboard.press(Key.right)

    prev_steering = steering_key

    if throttle_key != prev_throttle:
        if prev_throttle == Key.up:
            keyboard.release(Key.up)

        if throttle_key == Key.up:
            keyboard.press(Key.up)

    prev_throttle = throttle_key

    if brake_key != prev_brake:
        if prev_brake == Key.down:
            keyboard.release(Key.down)

        if brake_key == Key.down:
            keyboard.press(Key.down)

    prev_brake = brake_key


    cv2.imshow("Head Pose Racing Controls", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
