import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

NOSE_POINT = 1

def draw_3d_clown_nose(frame, x, y, radius):

    cv2.circle(frame, (x, y), radius, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (x - 10, y - 10), int(radius * 0.3), (255, 255, 255), -1, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[NOSE_POINT]
            h, w, _ = frame.shape
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)

            draw_3d_clown_nose(frame, nose_x, nose_y, 25)

    cv2.imshow("Webcam - Nez de Clown 3D", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
