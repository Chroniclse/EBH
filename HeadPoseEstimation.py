import cv2
import mediapipe as mp
import numpy as np
import serial
import time

arduino_port = "COM5"  
baudrate = 115200
try:
    arduino = serial.Serial(arduino_port, baudrate, timeout=1)
    time.sleep(2)  
except:
    print("Not connected")
    arduino = None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

def normalize(val, min_val, max_val):
    val = max(min_val, min(max_val, val))  # clamp
    return (val - min_val) / (max_val - min_val)

def send_haptic(command):
    if arduino:
        try:
            arduino.write((command + "\n").encode())
        except:
            pass

yaw_min, yaw_max = -10, 10      
pitch_min, pitch_max = -20, 20  
brake_threshold = -10            
offtrack_yaw_limit = 40          
offtrack_pitch_limit = 30        

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    start = time.time()
    img_h, img_w, img_c = image.shape

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    steering = 0
    throttle = 0
    brake = 0
    off_track = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d = []
            face_3d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    if idx == 1:
                        nose_2d = (x, y)
                        nose_3d = (x, y, lm.z * 3000)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = img_w
            cam_matrix = np.array([
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1]
            ])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d,
                                                       cam_matrix, dist_matrix)

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x_ang, y_ang, z_ang = angles * 360

   
            steering = 2 * normalize(y_ang, yaw_min, yaw_max) - 1   
            throttle = 1 - normalize(x_ang, pitch_min, pitch_max)  
            brake = 0
            if x_ang < brake_threshold:
                brake = normalize(-x_ang, 10, 20)  

            if abs(y_ang) > offtrack_yaw_limit or abs(x_ang) > offtrack_pitch_limit:
                off_track = True


            if off_track:
                send_haptic("OFF_TRACK")
            else:
                send_haptic("ON_TRACK")

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_ang * 10),
                  int(nose_2d[1] - x_ang * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            cv2.putText(image, f"Steer: {steering:.2f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Throttle: {throttle:.2f} Brake: {brake:.2f}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Off-track: {off_track}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if off_track else (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


    cv2.imshow("Head Pose Racing Controls", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
