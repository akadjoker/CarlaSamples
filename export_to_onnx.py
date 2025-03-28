import torch
from model_light import LightSteeringNet

 
MODEL_PATH = "model_light.pth"
ONNX_PATH = "model_light.onnx"

 
model = LightSteeringNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

 
dummy_input = torch.randn(1, 3, 64, 64)

 
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ Modelo exportado com sucesso para {ONNX_PATH}")

