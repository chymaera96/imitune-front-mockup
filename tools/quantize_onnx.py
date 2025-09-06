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

sess_fp32 = ort.InferenceSession("models/qvim.onnx")
sess_int8 = ort.InferenceSession("models/qvim.int8.onnx")

x = np.random.randn(1, 32000*10).astype("float32")  # 10s waveform
out_fp32 = sess_fp32.run(None, {"waveform": x})[0]
out_int8 = sess_int8.run(None, {"waveform": x})[0]

cos = np.dot(out_fp32.flatten(), out_int8.flatten()) / (
    np.linalg.norm(out_fp32.flatten())*np.linalg.norm(out_int8.flatten()) + 1e-12
)
print("Cosine similarity:", cos)