<<<<<<< HEAD
# pml10
### Parkinson-Test mit Fingerbewegung
## Teilnehmerin: 
Sarah Zalloukh
### Anwendung: 
Früherkennung motorischer Störungen wie Parkinson oder als Teil einer Parkinson-Vorsorge-App
Mit einer Webcam sollen feine Fingerbewegungen (z. B. Tippen von Daumen gegen Zeigefinger) erkannt und analysiert werden, um Hinweise auf motorische Einschränkungen (z. B. Zittern, Verlangsamung) zu geben.
=======
# ParkinsonDetektAI
ML gestützte Analyse von Fingerbewegungen zur Parkinson Früherkennung
________________________________________
📌 Über das Projekt

ParkinsonDetektAI ist ein Machine Learning basiertes System zur Analyse feiner Fingerbewegungen über eine einfache Webcam. Ziel ist die Früherkennung motorischer Störungen wie Parkinson, indem subtile Auffälligkeiten wie Tremor, Verlangsamung oder Unregelmäßigkeiten im Bewegungsablauf automatisch erkannt und quantifiziert werden.
________________________________________
🎯 Anwendungsbereiche

- Digitale Parkinson Vorsorge
- Telemedizinische Selbsttests
- Motorische Verlaufskontrolle
- Rehabilitation & Monitoring
- Forschung zu Bewegungsstörungen
________________________________________
🧠 Hauptfunktionen

-	Echtzeit Handerkennung mittels YOLO Pose
-	Fingertracking über Keypoints (Daumen bis kleiner Finger)
-	Tremorberechnung über Standardabweichung der Fingerpositionen
-	Visualisierung der Tremorintensität (grün / orange / rot)
-	Speicherung der Bewegungsdaten für spätere Analysen
________________________________________
🛠️ Technologien

-	YOLO Pose (Ultralytics)
-	OpenCV – Webcam Erfassung & Videobearbeitung
-	Matplotlib – Echtzeit Plots
-	Python – Hauptlogik & ML Pipeline
________________________________________
📁 Projektstruktur


```plaintext
├── HandDetektion.py            # YOLO-basierte Handerkennung
├── FingersDetection.py         # Finger-Keypoint-Erkennung
├── FingersTremorTracking.py    # Tremortracking & Analyse
├── TrainHandKeypoints.py       # Training des YOLO-Keypoint-Modells
├── models/                     # YOLO-Gewichte
├── data/                       # Testvideos, Keypoint-Daten
└── results/                    # Tremorplots & Analysen
```


________________________________________
📊 Ergebnisse

1. Kein Parkinson
-	Tremor stabilisiert sich schnell
-	Werte im Bereich 5–15 Einheiten
-	Ruhige, gesunde Motorik
2. Physiologischer Tremor
-	Mittlere Schwankungen (10–15 Einheiten)
-	Leichte Aktivität einzelner Finger
-	Kein krankhafter Tremor
3. Verdacht auf Parkinson
-	Starke Ausschläge (>240 Einheiten)
-	Dauerhaft unregelmäßige Bewegungen
-	Typisches Muster eines Ruhetremors
________________________________________
🚀 Ausblick

-	Integration in klinische Systeme
-	Frequenzanalyse & Tremorklassifikation
-	Mobile App für Selbsttests
-	Erweiterung auf weitere Symptome (Bradykinese, Rigor)





>>>>>>> 75b00e39fcf3c801db26da814c91920f8d86c401
