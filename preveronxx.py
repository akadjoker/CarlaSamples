import cv2
import numpy as np
import math
from PIL import Image

VIDEO_PATH = 'video_final.mp4'
ONNX_PATH = 'model_light.onnx'
IMG_SIZE = (64, 64)


net = cv2.dnn.readNetFromONNX(ONNX_PATH)


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

print("🎥 A correr com OpenCV DNN. Pressiona 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize(IMG_SIZE)
    img = np.array(pil).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  
    blob = img[np.newaxis, ...]  

    net.setInput(blob)
    output = net.forward()
    steering = float(output[0][0])

    # Indicador visual
    steering = max(-1.0, min(1.0, steering))
    angle = steering * math.radians(45)
    cx, cy, r = 100, 100, 40
    x = int(cx + r * math.sin(angle))
    y = int(cy - r * math.cos(angle))

    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.line(frame, (cx, cy), (x, y), (0, 255, 0), 3)

    cv2.imshow('Steering ONNX', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

