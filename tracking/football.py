import cv2
import numpy as np
from ultralytics import YOLO
import sys
sys.path.append('./sort')  
from sort import Sort 


model = YOLO('yolov8n.pt')

ball_tracker=Sort()
player_tracker = Sort()

# Abrir o vídeo
video_path = 'C:/Users/gabri/OneDrive/Documentos/python/vid.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    player_detections = []
    ball_detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == 0: 
                player_detections.append([x1, y1, x2, y2, conf])
            if cls == 32:  # 32: (bola)
                ball_detections.append([x1, y1, x2, y2, conf])

    #
    player_detections = np.array(player_detections) if len(player_detections) > 0 else np.empty((0, 5))
    ball_detections = np.array(ball_detections) if len(ball_detections) > 0 else np.empty((0, 5))

    player_tracks = player_tracker.update(player_detections)
    ball_tracks = ball_tracker.update(ball_detections)

    for track in player_tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caixa verde para jogadores
        cv2.putText(frame, f'Player {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for track in ball_tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Caixa vermelha para a bola
        cv2.putText(frame, f'Ball {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
