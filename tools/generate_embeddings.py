import os
import argparse
import json
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .export_onnx import InferenceWrapper
from qvim_mn_baseline.utils import NAME_TO_WIDTH


# ============================================================
# Dataset
# ============================================================

class AudioDataset(Dataset):
    def __init__(
        self,
        audio_dirs,
        tag_metadata_path,
        exclude_tag_list_path,
        sample_rate=32000,
        max_duration_sec=30.0,
        duration=10.0,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.fixed_length = int(sample_rate * duration)

        # -------------------------
        # Load metadata
        # -------------------------
        with open(tag_metadata_path, "r") as f:
            tag_metadata = json.load(f)

        with open(exclude_tag_list_path, "r") as f:
           exclude_tags = set(line.strip() for line in f if line.strip())


        audio_dirs = [os.path.abspath(d) for d in audio_dirs]

        def resolve_path(fname):
            if os.path.isabs(fname) and os.path.isfile(fname):
                return fname
            for d in audio_dirs:
                p = os.path.join(d, fname)
                if os.path.isfile(p):
                    return p
            return None

        # -------------------------
        # Filter files
        # -------------------------
        self.filepaths = []

        for fname, tags in tqdm(
            tag_metadata.items(),
            desc="Filtering audio (tags + duration)",
            total=len(tag_metadata),
        ):
            # Exclude by tags
            if any(tag in exclude_tags for tag in tags):
                continue

            fpath = resolve_path(fname)
            if fpath is None:
                continue

            # Exclude by duration
            try:
                dur = librosa.get_duration(path=fpath)
            except Exception:
                continue

            if dur > max_duration_sec:
                continue

            self.filepaths.append(fpath)


        print(f"[AudioDataset] Using {len(self.filepaths)} valid audio files")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]

        audio, _ = librosa.load(
            path,
            sr=self.sample_rate,
            mono=True,
            duration=self.duration,
        )

        if len(audio) < self.fixed_length:
            padded = np.zeros(self.fixed_length, dtype=np.float32)
            padded[:len(audio)] = audio
        else:
            padded = audio[:self.fixed_length]

        return torch.tensor(padded).float(), path


# ============================================================
# Embedding extraction
# ============================================================

def extract_embeddings(model, dataloader, device, emb_memmap):
    model.eval()
    model.to(device)

    offset = 0
    filepaths = []

    with torch.no_grad():
        for audio_batch, batch_paths in tqdm(dataloader, desc="Extracting embeddings"):
            audio_batch = audio_batch.to(device)

            output = model(audio_batch)  # (B, 960)
            output = output.cpu().numpy().astype(np.float32)

            B = output.shape[0]
            emb_memmap[offset: offset + B] = output
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
            outputs = outputs[0]

        assert outputs.ndim == 2, f"Expected output ndim 2, got {outputs.ndim}"

        B = outputs.shape[0]
        emb_memmap[offset: offset + B] = outputs
        offset += B
        filepaths.extend(batch_paths)

    emb_memmap.flush()
    return filepaths


# ============================================================
# Model loading
# ============================================================

def load_inference_model(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model = InferenceWrapper(cfg, state_dict)
    model.to(device)
    model.eval()
    return model


def get_infer_config(pretrained_name, sample_rate=32000, duration=10.0):
    from types import SimpleNamespace
    return SimpleNamespace(
        sample_rate=sample_rate,
        window_size=800,
        hop_size=320,
        n_fft=1024,
        n_mels=128,
        fmin=0,
        fmax=None,
        pretrained_name=pretrained_name,
        duration=duration,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio_dirs", nargs="+", required=True)
    parser.add_argument("--tag_metadata", type=str, default="tag_metadata.json")
    parser.add_argument("--exclude_tag_list", type=str, default="exclude_tag_list_80k.txt")

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--onnx_model", type=str, default="models/qvim.onnx")

    parser.add_argument("--pretrained_name", type=str, default="mn10_as")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--output_npy", type=str, default="/data/scratch/acw723/laion_fsd_excluded_embeddings_80k.npy")
    parser.add_argument("--output_metadata", type=str, default="laion_fsd_excluded_metadata_80k.csv")
    parser.add_argument("--freesound_meta_csv", type=str, default="freesound_meta.csv")

    args = parser.parse_args()

    config = get_infer_config(
        pretrained_name=args.pretrained_name,
        sample_rate=32000,
        duration=10.0,
    )

    dataset = AudioDataset(
        audio_dirs=args.audio_dirs,
        tag_metadata_path=args.tag_metadata,
        exclude_tag_list_path=args.exclude_tag_list,
        sample_rate=config.sample_rate,
        max_duration_sec=20.0,
        duration=config.duration,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    emb_dim = 960
    emb_memmap = np.memmap(
        args.output_npy,
        dtype="float32",
        mode="w+",
        shape=(len(dataset), emb_dim),
    )

    # -------------------------
    # Backend selection
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

        filepaths = extract_embeddings_onnx(
            session,
            dataloader,
            emb_memmap,
        )

    else:
        if args.checkpoint is None:
            raise ValueError("Either --checkpoint or --onnx_model must be provided")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_inference_model(args.checkpoint, config, device)

        filepaths = extract_embeddings(
            model,
            dataloader,
            device,
            emb_memmap,
        )

# ============================================================
# Post-extraction metadata join
# ============================================================

print("Loading freesound metadata...")
fs_meta = pd.read_csv(args.freesound_meta_csv)

# Build lookup: audio_filename -> freesound_url
fs_lookup = dict(
    zip(fs_meta["audio_filename"], fs_meta["freesound_url"])
)

output_rows = []
missing = 0

for fp in filepaths:
    # Normalize path
    fp_norm = os.path.normpath(fp)
    parts = fp_norm.split(os.sep)

    if len(parts) < 2:
        missing += 1
        continue

    parent_dir = parts[-2]          # train / test
    filename = parts[-1]             # filename.flac
    key = f"{parent_dir}/{filename}"

    freesound_url = fs_lookup.get(key)

    if freesound_url is None:
        missing += 1

    output_rows.append({
        "filepath": fp,
        "freesound_url": freesound_url,
    })

df_out = pd.DataFrame(output_rows)

df_out.to_csv(args.output_metadata, index=False)

print(f"Saved metadata to {args.output_metadata}")
print(f"Missing freesound_url for {missing} / {len(filepaths)} files")
