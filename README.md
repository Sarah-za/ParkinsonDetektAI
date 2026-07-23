# ParkinsonDetektAI
ParkinsonDetektAI
ML‑gestützte Analyse von Fingerbewegungen zur Parkinson‑Früherkennung

📌 Über das Projekt
ParkinsonDetektAI ist ein Machine‑Learning‑basiertes System zur Analyse feiner Fingerbewegungen über eine einfache Webcam.
Ziel ist die Früherkennung motorischer Störungen wie Parkinson, indem subtile Auffälligkeiten wie Tremor, Verlangsamung oder Unregelmäßigkeiten im Bewegungsablauf automatisch erkannt und quantifiziert werden.

🎯 Anwendungsbereiche
Digitale Parkinson‑Vorsorge

Telemedizinische Selbsttests

Motorische Verlaufskontrolle

Rehabilitation & Monitoring

Forschung zu Bewegungsstörungen

🧠 Hauptfunktionen
Echtzeit‑Handerkennung mittels YOLO‑Pose

Fingertracking über Keypoints (Daumen bis kleiner Finger)

Tremorberechnung über Standardabweichung der Fingerpositionen

Visualisierung der Tremorintensität (grün / orange / rot)

Speicherung der Bewegungsdaten für spätere Analysen

🛠️ Technologien
YOLO‑Pose (Ultralytics)

OpenCV – Webcam‑Erfassung & Videobearbeitung

Matplotlib – Echtzeit‑Plots

Python – Hauptlogik & ML‑Pipeline

📁 Projektstruktur
├── HandDetektion.py            # YOLO-basierte Handerkennung
├── FingersDetection.py         # Finger-Keypoint-Erkennung
├── FingersTremorTracking.py    # Tremortracking & Analyse
├── TrainHandKeypoints.py       # Training des YOLO-Keypoint-Modells
├── models/                     # YOLO-Gewichte
├── data/                       # Testvideos, Keypoint-Daten
└── results/                    # Tremorplots & Analysen
📊 Ergebnisse
1. Kein Parkinson
Tremor stabilisiert sich schnell

Werte im Bereich 5–15 Einheiten

Ruhige, gesunde Motorik

2. Physiologischer Tremor
Mittlere Schwankungen (10–15 Einheiten)

Leichte Aktivität einzelner Finger

Kein krankhafter Tremor

3. Verdacht auf Parkinson
Starke Ausschläge (>240 Einheiten)

Dauerhaft unregelmäßige Bewegungen

Typisches Muster eines Ruhetremors

🚀 Ausblick
Integration in klinische Systeme

Frequenzanalyse & Tremorklassifikation

Mobile App für Selbsttests

Erweiterung auf weitere Symptome (Bradykinese, Rigor)
