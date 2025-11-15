from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load Model Once
model = YOLO("../model/best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    results = model.predict(img, imgsz=640, conf=0.5)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = results[0].names[cls]
        detections.append(name)

    return {"ingredients": detections}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
