from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")

# Train with clear progress bar and logs
results = model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    device=0,      # GPU
    batch=16,
    verbose=True,  # <-- forces detailed output
)
