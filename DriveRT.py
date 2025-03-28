#!/usr/bin/env python3
import time
import threading
from Jetcar import JetCar
import cv2
import os
import datetime
import numpy as np
from PIL import Image
import math

 
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

IMG_SIZE = (64, 64)
ENGINE_PATH = "model_light.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def gstreamer_pipeline(
        capture_width=400,
        capture_height=400,
        display_width=640,
        display_height=480,
        framerate=30,
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


 
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    stream.synchronize()
    return [out[0] for out in outputs]


class Controller:
    def __init__(self):
        self.car = JetCar()
        self.car.start()
        time.sleep(0.5)

        self.steering = 0.0
        self.speed = 0.0
        self.max_speed = 0.7
        self.running = True

 
        self.engine = load_engine(ENGINE_PATH)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

        try:
            self.init_camera()
        except Exception as e:
            print(f"ERRO: {e}")
            exit(1)

    def init_camera(self):
        try:
            self.camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
            if not self.camera.isOpened():
                raise Exception("Falha ao abrir câmera")
            print("Câmera inicializada com sucesso")

            self.fps = self.camera.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
        except Exception as e:
            print(f"Erro ao inicializar câmera: {e}")
            self.camera = None

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

                # Pré-processamento
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize(IMG_SIZE)
                img = np.array(pil).astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1)).reshape(1, 3, 64, 64)
                np.copyto(self.inputs[0][0], img.ravel())

 
                output = do_inference(self.context, self.bindings, self.inputs, self.outputs, self.stream)
                self.steering = float(output[0])
                self.steering = max(-1.0, min(1.0, self.steering))

 
                self.car.set_steering(self.steering)
                self.process_frame(frame)

                key = cv2.waitKey(1)
                if key != -1:
                    if key == 27:
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

        cv2.rectangle(frame, (steering_bar_x, steering_bar_y),
                      (steering_bar_x + steering_bar_width, steering_bar_y + steering_bar_height),
                      (100, 100, 100), -1)

        center_x = steering_bar_x + steering_bar_width // 2
        indicator_pos_x = center_x + int(current_steering * (steering_bar_width // 2))
        cv2.rectangle(frame, (indicator_pos_x - 5, steering_bar_y - 5),
                      (indicator_pos_x + 5, steering_bar_y + steering_bar_height + 5), (0, 0, 255), -1)

        cv2.line(frame, (center_x, steering_bar_y - 5),
                 (center_x, steering_bar_y + steering_bar_height + 5), (255, 255, 255), 1)

        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            fps = 1 / (current_time - self.last_frame_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 100, 20), font, font_scale, (255, 255, 0), 1)
        self.last_frame_time = current_time

        cv2.imshow('Main', frame)


if __name__ == "__main__":
    controller = Controller()
    controller.run()
