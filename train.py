from ultralytics import YOLO
import clearml

model = YOLO('yolov8n.pt')

results = model.train(data='datasets/data.yaml', epochs=50, imgsz=640, batch=8)