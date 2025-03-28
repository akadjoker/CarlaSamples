import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import math
from PIL import Image

IMG_SIZE = (64, 64)
ENGINE_PATH = "model_light.engine"

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
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

def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize(IMG_SIZE)
    img = np.array(pil).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)).reshape(1, 3, 64, 64)  # NCHW
    return img

# ========== MAIN ==========

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

cap = cv2.VideoCapture("video_final.mp4")
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    input_image = preprocess(frame)
    np.copyto(inputs[0][0], input_image.ravel())

    output = do_inference(context, bindings, inputs, outputs, stream)
    steering = float(output[0])
    steering = max(-1.0, min(1.0, steering))
    angle = steering * math.radians(45)
    cx, cy, r = 100, 100, 40
    x = int(cx + r * math.sin(angle))
    y = int(cy - r * math.cos(angle))

    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.line(frame, (cx, cy), (x, y), (0, 255, 0), 3)
    cv2.imshow('Steering', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
