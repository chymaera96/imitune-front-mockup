#!/usr/bin/env python3
"""
Export your QViM Lightning checkpoint to ONNX for inference.
Run from the app repo root:

    python tools/export_onnx.py \
        --ckpt ../QBV2025/path/to/your.ckpt \
        --out models/qvim.onnx
"""

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# import modules from the training repo (installed via pip install git+https://github.com/chymaera96/QBV2025.git@triplet-mn)
from qvim_mbn_multi.utils import NAME_TO_WIDTH
from qvim_mbn_multi.mn.preprocess import AugmentMelSTFT
from qvim_mbn_multi.mn.model import get_model as get_mobilenet


class InferenceWrapper(torch.nn.Module):
    """Minimal wrapper: waveform -> mel -> encoder -> normalized embedding."""

    def __init__(self, cfg, state_dict):
        super().__init__()
        # mel front-end (disable augmentation)
        self.mel = AugmentMelSTFT(
            n_mels=cfg.n_mels,
            sr=cfg.sample_rate,
            win_length=cfg.window_size,
            hopsize=cfg.hop_size,
            n_fft=cfg.n_fft,
            freqm=0, timem=0,
            fmin=cfg.fmin, fmax=cfg.fmax,
            fmin_aug_range=0, fmax_aug_range=0
        )
        # encoder
        self.encoder = get_mobilenet(
            width_mult=NAME_TO_WIDTH(cfg.pretrained_name),
            pretrained_name=cfg.pretrained_name,
        )
        # load weights (ignore unmatched keys if PL saved extras)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print("Warning: missing keys:", missing)
        if unexpected:
            print("Warning: unexpected keys:", unexpected)

    def forward(self, x):
        # x: (B, T) waveform @ sample_rate
        mel = self.mel(x).unsqueeze(1)  # (B,1,n_mels,frames)
        out = self.encoder(mel)
        if isinstance(out, tuple):
            out = out[1]  # (cls, emb) -> emb
        return F.normalize(out, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--out", default="models/qvim.onnx", help="Output ONNX file")
    p.add_argument("--sample_rate", type=int, default=32000)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--window_size", type=int, default=800)
    p.add_argument("--hop_size", type=int, default=320)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--fmin", type=int, default=0)
    p.add_argument("--fmax", type=int, default=0)  # 0 = Nyquist in your code
    p.add_argument("--pretrained_name", default="mn10_as")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    # load Lightning checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    cfg = SimpleNamespace(**vars(args))
    model = InferenceWrapper(cfg, state_dict)
    model.eval()

    # dummy input: one 10s clip (batch=1)
    T = int(cfg.duration * cfg.sample_rate)
    dummy = torch.randn(1, T, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        export_params=True,
        opset_version=cfg.opset,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "samples"},
            "embedding": {0: "batch"},
        },
    )
    print("Exported ONNX model to", args.out)


if __name__ == "__main__":
    main()
