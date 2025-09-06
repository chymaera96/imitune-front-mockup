import os
import argparse
import torch
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from ex_qvim import QVIMModule  # assumes QVIMModule is in the same directory
from qvim_mbn_multi.mn.preprocess import AugmentMelSTFT
from qvim_mbn_multi.mn.model import get_model
from qvim_mbn_multi.utils import NAME_TO_WIDTH


class FSD50KDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=32000, duration=10.0):
        self.audio_paths = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".wav") or f.endswith(".mp3")
        ]
        self.sample_rate = sample_rate
        self.duration = duration
        self.fixed_length = int(self.sample_rate * self.duration)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True, duration=self.duration)
        if len(audio) < self.fixed_length:
            padded = np.zeros(self.fixed_length, dtype=np.float32)
            padded[:len(audio)] = audio
        else:
            padded = audio[:self.fixed_length]
        return torch.tensor(padded).float(), os.path.basename(path)


def extract_embeddings(model, dataloader, device):
    model.eval()
    model.to(device)

    embeddings = []
    filenames = []

    with torch.no_grad():
        for audio_batch, file_batch in tqdm(dataloader, desc="Extracting embeddings"):
            audio_batch = audio_batch.to(device)
            output = model.forward_pass(audio_batch)
            embeddings.append(output.cpu().numpy())
            filenames.extend(file_batch)

    return np.concatenate(embeddings), filenames


def load_model_from_ckpt(ckpt_path, config_args):
    model = QVIMModule(config_args)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    return model


def get_dummy_config(pretrained_name, sample_rate=32000, duration=10.0):
    from types import SimpleNamespace
    return SimpleNamespace(
        n_mels=128,
        sr=sample_rate,
        win_length=800,
        hop_size=320,
        n_fft=1024,
        freqm=2,
        timem=200,
        fmin=0,
        fmax=None,
        fmin_aug_range=10,
        fmax_aug_range=2000,
        pretrained_name=pretrained_name,
        initial_tau=0.07,
        tau_trainable=False,
        margin=0.2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to directory with audio files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained .ckpt file")
    parser.add_argument("--pretrained_name", type=str, default="mn10_as", help="Width multiplier name for MobileNet")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_npy", type=str, default="fsd50k_embeddings.npy")
    args = parser.parse_args()

    # Config & Dataset
    config = get_dummy_config(pretrained_name=args.pretrained_name)
    dataset = FSD50KDataset(args.audio_dir, sample_rate=config.sr, duration=config.duration)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_ckpt(args.checkpoint, config)

    # Extract and save
    embeddings, filenames = extract_embeddings(model, dataloader, device)
    np.save(args.output_npy, embeddings)
    print(f"Saved embeddings of shape {embeddings.shape} to {args.output_npy}")
