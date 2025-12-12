import os
import argparse
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader

# from qvim_mn_baseline.ex_qvim import QVIMModule
from .export_onnx import InferenceWrapper
from qvim_mn_baseline.utils import NAME_TO_WIDTH


class AudioDataset(Dataset):
    def __init__(self, audio_dirs, sample_rate=32000, duration=10.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.fixed_length = int(sample_rate * duration)
        self.filepaths = []

        # for audio_dir in audio_dirs:
        #     for fname in os.listdir(audio_dir):
        #         if fname.endswith(".wav") or fname.endswith(".mp3"):
        #             self.filepaths.append(os.path.join(audio_dir, fname))
        for fpath in glob.glob(os.path.join(*audio_dirs, '**', '*.*'), recursive=True):
            if fpath.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                self.filepaths.append(fpath)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True, duration=self.duration)

        if len(audio) < self.fixed_length:
            padded = np.zeros(self.fixed_length, dtype=np.float32)
            padded[:len(audio)] = audio
        else:
            padded = audio[:self.fixed_length]

        return torch.tensor(padded).float(), path



def extract_embeddings(model, dataloader, device, emb_memmap):
    model.eval()
    model.to(device)

    offset = 0
    filepaths = []

    with torch.no_grad():
        for audio_batch, batch_paths in tqdm(dataloader, desc="Extracting embeddings"):
            audio_batch = audio_batch.to(device)

            output = model(audio_batch)              # (B, 960)
            output = output.cpu().numpy().astype(np.float32)

            B = output.shape[0]
            emb_memmap[offset : offset + B] = output
            offset += B

            filepaths.extend(batch_paths)

    emb_memmap.flush()
    return filepaths

def extract_embeddings_onnx(session, dataloader, emb_memmap):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    offset = 0
    filepaths = []

    for audio_batch, batch_paths in tqdm(dataloader, desc="Extracting ONNX embeddings"):
        audio_np = audio_batch.numpy().astype(np.float32)
        outputs = session.run([output_name], {input_name: audio_np})[0]

        if outputs.ndim == 3 and outputs.shape[0] == 1:
            outputs = outputs[0]  # (B,960)

        assert outputs.ndim == 2, f"Expected output ndim 2, got {outputs.ndim}"
        B = outputs.shape[0]
        emb_memmap[offset : offset + B] = outputs
        offset += B
        filepaths.extend(batch_paths)

    emb_memmap.flush()
    return filepaths


def load_inference_model(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # most PL ckpts store weights under 'state_dict'
    state_dict = ckpt.get('state_dict', ckpt)
    model = InferenceWrapper(cfg, state_dict)
    model.to(device)
    model.eval()
    return model


def get_infer_config(pretrained_name, sample_rate=32000, duration=10.0):
    from types import SimpleNamespace
    return SimpleNamespace(
        # names expected by InferenceWrapper
        sample_rate=sample_rate,
        window_size=800,
        hop_size=320,
        n_fft=1024,
        n_mels=128,
        fmin=0,
        fmax=None,
        pretrained_name=pretrained_name,
        # also keep duration around for dataset construction
        duration=duration
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dirs", nargs='+', required=True, help="List of directories with audio files")
    parser.add_argument("--checkpoint", type=str, help="Path to pretrained .ckpt file")
    parser.add_argument("--pretrained_name", type=str, default="mn10_as", help="Width multiplier name for MobileNet")
    parser.add_argument("--onnx_model", type=str, help="Path to ONNX model (ONNX mode)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_npy", type=str, default="fsd50k_embeddings.npy")
    parser.add_argument("--output_metadata", type=str, default="fsd50k_metadata.csv")
    args = parser.parse_args()

    # Setup
    config = get_infer_config(pretrained_name=args.pretrained_name, sample_rate=32000, duration=10.0)
    dataset = AudioDataset(audio_dirs=args.audio_dirs, sample_rate=config.sample_rate, duration=config.duration)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    emb_dim = 960
    emb_memmap = np.memmap(
        args.output_npy,
        dtype='float32',
        mode='w+',
        shape=(len(dataset), emb_dim) 
    )


    # -------------------------
    # Choose backend
    # -------------------------
    if args.onnx_model is not None:
        import onnxruntime as ort

        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]

        session = ort.InferenceSession(
            args.onnx_model,
            providers=providers,
        )

        # input_name = session.get_inputs()[0].name
        # output_name = session.get_outputs()[0].name

        # audio_np = np.random.randn(12, 32000 * 10).astype(np.float32)
        # out = session.run([output_name], {input_name: audio_np})[0]
        # assert out.shape == (12, 960), f"Expected output shape (12, 960), got {out.shape}"


        filepaths = extract_embeddings_onnx(session, dataloader, emb_memmap)

    else:
        if args.checkpoint is None:
            raise ValueError("Either --checkpoint or --onnx_model must be provided")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_inference_model(args.checkpoint, config, device)

        # assert model(torch.randn(16, config.sample_rate * config.duration)).shape[0] == 16, \
        #     f"Model forward pass failed. Expected batch size 16 output, got {model(torch.randn(16, config.sample_rate * config.duration)).shape[0]}."  

        filepaths = extract_embeddings(model, dataloader, device, emb_memmap)

    # -------------------------
    # Save outputs
    # -------------------------
    pd.DataFrame({"filepath": filepaths}).to_csv(
        args.output_metadata, index=False
    )

    print(f"Saved embeddings to {args.output_npy}")
    print(f"Saved metadata to {args.output_metadata}")
    print(f"Saved metadata to {args.output_metadata}")


