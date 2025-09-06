import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import torch

quantize_dynamic(
    model_input="models/qvim.onnx",
    model_output="models/qvim.int8.onnx",
    weight_type=QuantType.QInt8,   # use 8-bit signed
    per_channel=True,              # (optional) finer quantization granularity
    op_types_to_quantize=["MatMul","Gemm"]  # <- avoid convs
)
print("Wrote quantized model to models/qvim.int8.onnx")

# import numpy as np, onnxruntime as ort

sess_fp32 = ort.InferenceSession("models/qvim.onnx", providers=["CPUExecutionProvider"])
sess_int8 = ort.InferenceSession("models/qvim.int8.onnx", providers=["CPUExecutionProvider"])

x = np.random.randn(1, 32000*10).astype("float32")
y0 = sess_fp32.run(None, {"waveform": x})[0]
y1 = sess_int8.run(None, {"waveform": x})[0]

cos = (y0.flatten() @ y1.flatten()) / (np.linalg.norm(y0)*np.linalg.norm(y1) + 1e-12)
print("cosine similarity:", cos)
