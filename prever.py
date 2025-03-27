import cv2
import torch
import numpy as np
from torchvision import transforms
from model_light import LightSteeringNet
from PIL import Image
import math

# ============================
# Configurações
# ============================
MODEL_PATH = 'model_light.pth'
VIDEO_PATH = 'video_final.mp4'
IMG_SIZE = (64, 64)

# ============================
# Transform
# ============================
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# ============================
# Modelo
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightSteeringNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
for param in model.parameters():
    param.requires_grad = False
print(model.parameters)

# ============================
# Video
# ============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        steering = model(tensor).item()

    # ============================
    # Indicador de direção visual
    # ============================
    steering = max(-1.0, min(1.0, steering))
    angle = steering * math.radians(45)
    cx, cy, r = 100, 100, 40
    x = int(cx + r * math.sin(angle))
    y = int(cy - r * math.cos(angle))

    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.line(frame, (cx, cy), (x, y), (0, 255, 0), 3)

    cv2.imshow('Steering ', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

