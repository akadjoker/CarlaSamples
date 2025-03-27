import torch
from model_light import LightSteeringNet

# ============================
# Configurações
# ============================
MODEL_PATH = 'model_light.pth'
ONNX_PATH = 'model_light.onnx'
IMG_SIZE = (64, 64)

# ============================
# Modelo
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightSteeringNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================
# Exportação
# ============================
dummy_input = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['steering'],
    dynamic_axes={'input': {0: 'batch_size'}, 'steering': {0: 'batch_size'}}
)

print(f"Modelo exportado para {ONNX_PATH}")
