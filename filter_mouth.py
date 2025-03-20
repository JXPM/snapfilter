import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Webcam - Rouge à Lèvres Réaliste", cv2.WINDOW_NORMAL)

outer_lips_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
inner_lips_landmarks = [78, 95, 88, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

lip_hue = 170  
lip_saturation = 200  
alpha = 0.75 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            outer_lips_coords = np.array([
                (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
                for idx in outer_lips_landmarks
            ], np.int32)

            inner_lips_coords = np.array([
                (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
                for idx in inner_lips_landmarks
            ], np.int32)

            if len(outer_lips_coords) > 0 and len(inner_lips_coords) > 0:
                lips_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(lips_mask, [cv2.convexHull(outer_lips_coords)], 255) 
                cv2.fillPoly(lips_mask, [cv2.convexHull(inner_lips_coords)], 0)   

                lips_mask = cv2.GaussianBlur(lips_mask, (5, 5), 3)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                hsv[..., 0] = np.where(lips_mask == 255, lip_hue, hsv[..., 0])  
                hsv[..., 1] = np.where(lips_mask == 255, lip_saturation, hsv[..., 1])  

                colored_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                frame = cv2.addWeighted(frame, 1 - alpha, colored_frame, alpha, 0)
                
                cv2.polylines(frame, [outer_lips_coords], isClosed=True, color=(0, 0, 255), thickness=1)


    cv2.imshow("Webcam - Rouge à Lèvres Réaliste", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
