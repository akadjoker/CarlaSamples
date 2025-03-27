#!/usr/bin/env python3
import time
import threading
from Jetcar import JetCar
import cv2
import os
import datetime

import torch
import numpy as np
from model_light import LightSteeringNet
from PIL import Image
import math

IMG_SIZE = (64, 64)

def transform(pil_img):
    pil_img = pil_img.resize(IMG_SIZE)
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    return torch.tensor(img_array, dtype=torch.float32)

def gstreamer_pipeline(
        capture_width=128,# os minimos .. vamos ver
        capture_height=128,
        display_width=128,
        display_height=128,
        framerate=15,
        flip_method=0,
    ):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (flip_method, display_width, display_height)
    )

class Controller:
    def __init__(self):
        self.car = JetCar()
        self.car.start()
        time.sleep(0.5)
        
        self.steering = 0.0
        self.speed = 0.0
        self.max_speed = 0.7

        self.running = True
        self.show_gui = True  # coloca False se não quiseres janela com vídeo
        self.skip_frames = 2  # processa 1 a cada 3 frames
        self.frame_count = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LightSteeringNet().to(self.device)
        self.model.load_state_dict(torch.load('model_light.pth', map_location=self.device))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        try:
            self.init_camera()
        except Exception as e:
            print(f"ERRO: {e}")
            exit(1)

    def init_camera(self):
        self.camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.camera.isOpened():
            raise Exception("Falha ao abrir câmera")
        print("Câmera inicializada com sucesso")

    def handle_keyboard(self, key):
        key_char = chr(key & 0xFF).lower()
 

    def run(self):
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    time.sleep(0.1)
                    continue

                self.frame_count += 1
                process_this = (self.frame_count % (self.skip_frames + 1) == 0)

                if process_this:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    tensor = transform(pil).unsqueeze(0).to(self.device)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    tensor = transform(pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        self.steering = self.model(tensor).item()

                    self.steering = max(-1.0, min(1.0, self.steering))
                    self.car.set_steering(self.steering)

                if self.show_gui:
                    self.process_frame(frame)

                key = cv2.waitKey(1)
                if key != -1:
                    if key == 27:  # ESC
                        print("\nSaindo...")
                        break
                    else:
                        self.handle_keyboard(key)

        except KeyboardInterrupt:
            print("\nPrograma interrompido")
        finally:
            self.car.set_speed(0)
            self.car.set_steering(0)
            self.car.stop()

            if self.camera:
                self.camera.release()

            cv2.destroyAllWindows()
            print("ByBy!")

    def process_frame(self, frame):
        current_steering = self.steering

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 0)
        font_thickness = 2

        steering_text = f"Direcao: {current_steering:.2f}"
        cv2.putText(frame, steering_text, (10, 60), font, font_scale, font_color, font_thickness)

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        steering_bar_width = 200
        steering_bar_height = 20
        steering_bar_x = frame_width - steering_bar_width - 10
        steering_bar_y = 30

        cv2.rectangle(frame,
                    (steering_bar_x, steering_bar_y),
                    (steering_bar_x + steering_bar_width, steering_bar_y + steering_bar_height),
                    (100, 100, 100), -1)

        center_x = steering_bar_x + steering_bar_width // 2
        indicator_pos_x = center_x + int(current_steering * (steering_bar_width // 2))
        cv2.rectangle(frame,
                    (indicator_pos_x - 5, steering_bar_y - 5),
                    (indicator_pos_x + 5, steering_bar_y + steering_bar_height + 5),
                    (0, 0, 255), -1)

        cv2.line(frame,
                (center_x, steering_bar_y - 5),
                (center_x, steering_bar_y + steering_bar_height + 5),
                (255, 255, 255), 1)

        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            fps = 1 / (current_time - self.last_frame_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 100, 20), font, font_scale, (255, 255, 0), 1)
        self.last_frame_time = current_time

        cv2.imshow('Main', frame)

if __name__ == "__main__":
    controller = Controller()
    controller.run()

