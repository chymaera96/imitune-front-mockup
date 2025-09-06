import os
import argparse
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# from qvim_mn_baseline.ex_qvim import QVIMModule
from tools.export_onnx import InferenceWrapper
from qvim_mn_baseline.utils import NAME_TO_WIDTH


class AudioDataset(Dataset):
    def __init__(self, audio_dirs, sample_rate=32000, duration=10.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.fixed_length = int(sample_rate * duration)
        self.filepaths = []

        for audio_dir in audio_dirs:
            for fname in os.listdir(audio_dir):
                if fname.endswith(".wav") or fname.endswith(".mp3"):
                    self.filepaths.append(os.path.join(audio_dir, fname))

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



def extract_embeddings(model, dataloader, device):
    model.eval()
    model.to(device)

    embeddings = []
    filepaths = []

    with torch.no_grad():
        for audio_batch, batch_paths in tqdm(dataloader, desc="Extracting embeddings"):
            audio_batch = audio_batch.to(device)
            output = model.forward(audio_batch)
            embeddings.append(output.cpu().numpy())
            filepaths.extend(batch_paths)

    return np.concatenate(embeddings), filepaths



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
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained .ckpt file")
    parser.add_argument("--pretrained_name", type=str, default="mn10_as", help="Width multiplier name for MobileNet")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_npy", type=str, default="fsd50k_embeddings.npy")
    parser.add_argument("--output_metadata", type=str, default="fsd50k_metadata.csv")
    args = parser.parse_args()

    # Setup
    config = get_infer_config(pretrained_name=args.pretrained_name, sample_rate=32000, duration=10.0)
    dataset = AudioDataset(audio_dirs=args.audio_dirs, sample_rate=config.sample_rate, duration=config.duration)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_inference_model(args.checkpoint, config, device)

    # Extract embeddings
    embeddings, filepaths = extract_embeddings(model, dataloader, device)


    # Save
    np.save(args.output_npy, embeddings)
    pd.DataFrame({'filepath': filepaths}).to_csv(args.output_metadata, index=False)

    print(f"Saved embeddings to {args.output_npy}")
    print(f"Saved metadata to {args.output_metadata}")


