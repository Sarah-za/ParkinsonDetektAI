from ultralytics import YOLO

# Load a model
# model = YOLO("yolo/yolo11n-pose.pt") # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)

# Load the partially trained model
model = YOLO("yolo/last.pt")

# Resume training
results = model.train(resume=True)