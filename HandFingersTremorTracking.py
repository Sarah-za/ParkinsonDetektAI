import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Modell laden
model = YOLO("yolo/last.pt")  # dein Hand-Pose-Modell

# Webcam öffnen
cap = cv2.VideoCapture(0)

# Liste der Keypoint-Indizes, z. B. Fingerspitzen
finger_indices = [4, 8, 12, 16, 20]

# Bewegungshistorien für x und y getrennt
history_x = defaultdict(list)
history_y = defaultdict(list)
max_len = 30  # Wie viele Frames zur Analyse

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Pose-Inferenz
    results = model.predict(source=frame, save=False, conf=0.5, verbose=False)
    if not results or not results[0].keypoints or results[0].keypoints.shape[1] == 0:
        cv2.imshow("Tremor Tracking", frame)
        if cv2.waitKey(1) == 27: break
        continue

    keypoints = results[0].keypoints.xy[0].cpu().numpy()

    for i, index in enumerate(finger_indices):
        if index >= len(keypoints): continue
        x, y = keypoints[index]

        history_x[i].append(x)
        history_y[i].append(y)

        if len(history_x[i]) > max_len:
            history_x[i].pop(0)
            history_y[i].pop(0)

        std_x = np.std(history_x[i])
        std_y = np.std(history_y[i])
        tremor = np.sqrt(std_x**2 + std_y**2)

        # Farbe je nach Stärke
        if tremor < 2:
            color = (0, 255, 0)  # Grün = ruhig
        elif tremor < 5:
            color = (0, 165, 255)  # Orange = mittel
        else:
            color = (0, 0, 255)  # Rot = starkes Zittern

        # Punkte & Texte anzeigen
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        cv2.putText(frame,
                    f"F{i} X:{std_x:.1f} Y:{std_y:.1f} T:{tremor:.1f}",
                    (int(x) + 5, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2)

    cv2.imshow("Tremor Tracking", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break
#„s“ drückst, wird ein Bild als JPG zu speichern
if cv2.waitKey(1) & 0xFF == ord('s'):
    filename = f"frame_{cv2.getTickCount()}.jpg"
    cv2.imwrite(filename, frame)

# Load the model and run inference on the webcam
YOLO('yolo/last.pt').predict(source=0, show=True, save=True, project='runs/pose/output', name='tests')
cap.release()
cv2.destroyAllWindows()
