import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import matplotlib.pyplot as plt



# Modell laden
model = YOLO("yolo/last.pt")  # dein Hand-Pose-Modell

# Webcam öffnen
cap = cv2.VideoCapture(0)
# VideoWriter vorbereiten
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = "C:/Users/sarah/Desktop/PML/pml10/runs/pose/output/"
filename = f"{output_path}tremor_{int(time.time())}.avi"
out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

# Liste der Keypoint-Indizes, z. B. Fingerspitzen
finger_indices = [4, 8, 12, 16, 20]

# Bewegungshistorien für x und y getrennt
history_x = defaultdict(list)
history_y = defaultdict(list)
max_len = 30  # Wie viele Frames zur Analyse
# Testdauer (z. B. 10 Sekunden)
test_duration = 30  # oder dein gewünschter Wert
start_time = time.time()
print(f"🕒 Tremor-Test gestartet. Dauer: {test_duration} Sekunden.")

# Tremor-Werte speichern
tremor_values = defaultdict(list)

start_time = time.time()
time_stamps = defaultdict(list)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Pose-Inferenz
    results = model.predict(source=frame, save=False, conf=0.5, verbose=False)
    if not results or not results[0].keypoints or results[0].keypoints.shape[1] == 0:
        cv2.imshow("Tremor Tracking", frame)
        out.write(frame)
        if cv2.waitKey(1) == 27: break
        continue

    keypoints = results[0].keypoints.xy[0].cpu().numpy()

    for i, index in enumerate(finger_indices):
        if index >= len(keypoints): continue
        x, y = keypoints[index]

        history_x[i].append(x)
        history_y[i].append(y)
        #elapsed_time = time.time() - start_time
        # tremor_values[i].append(tremor)
        # time_stamps[i].append(elapsed_time)



        if len(history_x[i]) > max_len:
            history_x[i].pop(0)
            history_y[i].pop(0)

        std_x = np.std(history_x[i])
        std_y = np.std(history_y[i])
        tremor = np.sqrt(std_x**2 + std_y**2)
        tremor_values[i].append(tremor)
        elapsed_time = time.time() - start_time
        time_stamps[i].append(elapsed_time)


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
    out.write(frame)

    if time.time() - start_time > test_duration:
        print("✅ Testdauer erreicht.")
        break

    if cv2.waitKey(1) == 27:  # ESC
        break
cap.release()
out.release()
cv2.destroyAllWindows()
# Nach dem Test: einfache Entscheidung
all_tremor = []
for values in tremor_values.values():
    if values:
        avg = np.mean(values)
        all_tremor.append(avg)

if all_tremor:
    overall_avg_tremor = np.mean(all_tremor)
    print(f"🔎 Durchschnittlicher Tremor: {overall_avg_tremor:.2f}")
    if overall_avg_tremor > 20:  # Schwellwert anpassen
        print("⚠️ Verdacht auf Parkinson (starkes Zittern erkannt).")
    else:
        print("✅ Kein Hinweis auf Parkinson (Tremor im normalen Bereich).")
else:
    print("⚠️ Keine ausreichend Daten für Analyse.")
plt.figure(figsize=(10, 6))
for i in tremor_values.keys():
    times = time_stamps[i]
    tremors = tremor_values[i]
    if len(times) == len(tremors):  # Sicherstellen gleiche Länge
        plt.plot(times, tremors, label=f'Finger {i}')

plt.ylabel('Tremor')
plt.title('Tremor über Zeit')

plt.title('Fingerbewegung über Zeit')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_path}tremor_plot_{int(time.time())}.png")
plt.show()
