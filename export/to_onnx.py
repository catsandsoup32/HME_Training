import torch
import torch.onnx
from models import Model_1

import os
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

device = 'cuda'


tgt_in = torch.ones([1, 1], dtype=torch.long).to(device) # Changed from torch.long
tgt_mask = torch.triu(torch.ones(1, 1) * float("-inf"), diagonal=1).to(device)
#tgt_mask = torch.triu(torch.ones(1, 1, dtype=torch.bool), diagonal=1).to(device)

torch_model = Model_1(vocab_size=217, d_model=256, nhead=8, dim_FF=1024, dropout=0.3, num_layers=3).to(device)
#torch_model.load_state_dict(torch.load('runs/Exp8E8End_Acc=0.6702922582626343.pt', weights_only=True))
torch_model.eval()
torch_input = (torch.randn(1, 3, 384, 512).to(device), 
               tgt_in.to(device), 
               tgt_mask)

class MyModel(torch.nn.Module):
        def init(self) -> None:
            super().init()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x, bias=None):
            out = self.linear(x)
            out = out + bias
            return out


hopeless_model = MyModel()
kwargs = {"bias": 3.0}
args = (torch.randn(2, 2, 2),)
onnx_program = torch.onnx.dynamo_export(hopeless_model, *args, **kwargs).save(
    "my_simple_model.onnx"
)



'''
torch.onnx.export(
    torch_model, 
    torch_input,
    "model_1.onnx",
    export_params = True,
    input_names = ["image", "tgt", "tgt_mask"],
    output_names = ["output1"],
    dynamic_axes = {
        "tgt": {1: "tgt_axis_1"},
        "tgt_mask": {0: "tgt_mask_0", 1: "tgt_mask_1"},
        "output1": {1: "output1"}
    }
)
'''


import onnx

model = onnx.load("./model_1.onnx")
print([input.name for input in model.graph.input])

onnx.checker.check_model(model)
inferred_model = onnx.shape_inference.infer_shapes(model)
# print(onnx.helper.printable_graph(inferred_model.graph))

import onnxruntime as ort
import numpy as np
from PIL import Image

onnx_model_path = "model_1.onnx"
session = ort.InferenceSession(onnx_model_path)

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 384))
    image_array = np.array(image).transpose(2, 0, 1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

test_image = load_and_preprocess_image('test3.png').astype(np.float32)
test_tgt = np.ones((1, 2), dtype=np.int64)
test_mask = np.triu(np.ones((2, 2), dtype=np.float32) * float('-inf'), k=1)

inputs = {"image": test_image, 
          "tgt": test_tgt, 
          "tgt_mask": test_mask
          }

outputs = session.run(None, inputs)
outputs = outputs[0][0][0]
max_val = max(outputs)
print(max_val)
print(np.where(outputs == max_val))