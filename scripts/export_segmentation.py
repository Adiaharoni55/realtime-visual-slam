import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

MODEL_PATH = "models/deeplabv3_mobilenet.onnx"
IMAGE_PATH = "data/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846321.637163.png"
PERSON_CLASS = 15

class OutOnly(torch.nn.Module):
    """torchvision returns a dict; ONNX export needs a plain tensor."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]

# 1. Load the pretrained model, in eval mode
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights)
wrapped = OutOnly(model).eval()

# 2. Export to ONNX (dummy input 1x3x480x640: one RGB image at TUM's resolution)
dummy = torch.zeros(1, 3, 480, 640)
torch.onnx.export(wrapped, dummy, MODEL_PATH,
                  input_names=["input"], output_names=["output"],
                  opset_version=18, external_data=False)
print("Saved", MODEL_PATH)

# 3. Preprocess the real image exactly as the model expects
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))              # 480x640x3, 0-255
x = image.astype(np.float32) / 255.0
x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)              # 1x3x480x640

# 4. The same input through PyTorch and ONNX Runtime must give the same output
with torch.no_grad():
    out_torch = wrapped(x).numpy()
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
out_onnx = session.run(["output"], {"input": x.numpy()})[0]

classes_torch = out_torch.argmax(axis=1)[0]                          # 480x640, class per pixel
classes_onnx = out_onnx.argmax(axis=1)[0]
print("Max absolute difference:", np.abs(out_torch - out_onnx).max())
print("Same class: %.4f%% of pixels" % (100.0 * (classes_torch == classes_onnx).mean()))

# 5. Overlay the person mask in red
mask = classes_onnx == PERSON_CLASS
print("Person pixels: %.1f%%" % (100.0 * mask.mean()))
overlay = image.copy()
overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
Image.fromarray(overlay).save("output/person_mask.png")
print("Saved output/person_mask.png")

