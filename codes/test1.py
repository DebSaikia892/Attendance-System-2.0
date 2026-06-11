import torch
from ultralytics import YOLO
import cv2
from insightface.app import FaceAnalysis

device = "cuda" if torch.cuda.is_available() else "cpu"
#YOLOv8n
detector = YOLO("yolov8n-face.pt")
detector.to(device)
#ArcFace
# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=0) # GPU

#WebCam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    frame = cv2.resize(frame, (640, 480))

    #YOLOv8n 
    results = detector(frame, verbose=False)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            #Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("YOLOv8n", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

