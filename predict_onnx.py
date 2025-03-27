import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import math

 
MODEL_PATH = 'steering_model_light.onnx'
#VIDEO_PATH = '/home/djoker/code/Seame/sea:me/session_20250324_170653/video_20250324_172157_whatsapp.mp4'
VIDEO_PATH = 'video_final.mp4'
IMG_SIZE = (64, 64)

# ============================
# Modelo ONNX
# ============================
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# ============================
# Video
# ============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v', 'XVID'
out = cv2.VideoWriter("model.mp4", fourcc, 30, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
        #break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize(IMG_SIZE)
    img_np = np.array(pil).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # CxHxW
    img_np = np.expand_dims(img_np, axis=0)  # batch

    steering = session.run(None, {input_name: img_np})[0].squeeze()

 
    steering = max(-1.0, min(1.0, steering))
    angle = steering * math.radians(45)
    cx, cy, r = 100, 100, 40
    x = int(cx + r * math.sin(angle))
    y = int(cy - r * math.cos(angle))

    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.line(frame, (cx, cy), (x, y), (0, 255, 0), 3)

    cv2.imshow('Steering', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

