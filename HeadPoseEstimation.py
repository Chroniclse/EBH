import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    start = time.time()

    # Flip + convert
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            lm = face_landmarks.landmark

            # -------- EYE-BASED LEFT/RIGHT DETECTION --------
            left_eye_corner = (lm[33].x * img_w, lm[33].y * img_h)
            right_eye_corner = (lm[263].x * img_w, lm[263].y * img_h)

            eye_mid_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
            face_center_x = img_w / 2

            eye_offset = eye_mid_x - face_center_x

            left_right_threshold = 40  # px threshold

            # Convert eye_offset → yaw degrees (scaled)
            max_yaw_angle = 35  # realistic head rotation
            yaw_deg = (eye_offset / (img_w / 2)) * max_yaw_angle


            # -------- 2D POINTS FOR PnP PITCH --------
            image_points = np.array([
                (lm[1].x   * img_w, lm[1].y   * img_h),   # Nose tip
                (lm[199].x * img_w, lm[199].y * img_h),   # Chin
                (lm[33].x  * img_w, lm[33].y  * img_h),   # Left eye
                (lm[263].x * img_w, lm[263].y * img_h),   # Right eye
                (lm[61].x  * img_w, lm[61].y  * img_h),   # Left mouth
                (lm[291].x * img_w, lm[291].y * img_h),   # Right mouth
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
                object_points,
                image_points,
                cam_matrix,
                dist_matrix,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch_deg = angles[0] * 180  # degrees


            # -------- HEAD MOVEMENT LOGIC --------
            
            if pitch_deg > 15:
                text = "Forward"
            if eye_offset < -left_right_threshold:
                text = "Left"
            elif eye_offset > left_right_threshold:
                text = "Right"
            elif pitch_deg < -25:
                text = "Brake"
            else:
                text = "Forward"

            # ---------- DISPLAY ANGLES ----------
            cv2.putText(image, f"Yaw: {yaw_deg:.1f}°", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            cv2.putText(image, f"Pitch: {pitch_deg:.1f}°", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

            cv2.putText(image, text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # FPS
            fps = 1 / (time.time() - start)
            cv2.putText(image, f'FPS: {int(fps)}', (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    cv2.imshow("Head Pose Estimation", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
