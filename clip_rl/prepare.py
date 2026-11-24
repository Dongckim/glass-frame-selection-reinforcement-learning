"""Cache independent OpenCLIP reward and MobileNet policy features."""

import json
import subprocess
import tempfile
from itertools import batched
from pathlib import Path

from .data import Frame, Source, cache_path, fingerprint, read_frames, read_source


def video_geometry(source: Source, frames: list[Frame]) -> tuple[int, int]:
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
        "-show_entries", "stream=width,height:frame=best_effort_timestamp_time",
        "-of", "json", str(source.video),
    ])
    metadata = json.loads(probe)
    timestamps = metadata["frames"]
    for frame in frames:
        if (
            frame.index >= len(timestamps)
            or abs(float(timestamps[frame.index]["best_effort_timestamp_time"]) - frame.video_time) > 0.002
        ):
            raise ValueError(f"{source.video}: frame {frame.index} does not match the CSV timestamp.")
    stream = metadata["streams"][0]
    return stream["width"], stream["height"]


def decode_frames(source: Source, frames: list[Frame]):
    from PIL import Image

    width, height = video_geometry(source, frames)
    start, end = frames[0].index, frames[-1].index + 1
    wanted = {frame.index for frame in frames}
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-noautorotate", "-i", str(source.video),
        "-map", "0:v:0", "-vf", f"trim=start_frame={start}:end_frame={end}",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-threads", "1", "pipe:1",
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        try:
            for index in range(start, end):
                pixels = process.stdout.read(width * height * 3)
                if len(pixels) != width * height * 3:
                    raise RuntimeError(f"{source.video}: decoder ended before frame {index}.")
                if index in wanted:
                    yield Image.frombytes("RGB", (width, height), pixels)
            if process.wait() != 0:
                raise RuntimeError(f"{source.video}: ffmpeg failed.")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def prepare(annotations, cache_dir: Path, batch_size: int, device: str) -> None:
    import numpy as np
    import open_clip
    import torch
    from torch.nn.functional import normalize
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    clip_model, _, clip_transform = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device,
    )
    clip_model.eval()
    tokenize = open_clip.get_tokenizer("ViT-B-32")
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    mobile = mobilenet_v3_small(weights=weights).eval().to(device)
    mobile_transform = weights.transforms()

    with torch.inference_mode():
        for annotation in annotations:
            source = read_source(annotation)
            frames = read_frames(source)
            signature = fingerprint(source)
            clip_question = normalize(clip_model.encode_text(tokenize([source.question]).to(device)).float())
            clip_rows, mobile_rows = [], []
            for images in batched(decode_frames(source, frames), batch_size):
                clip_input = torch.stack([clip_transform(image) for image in images]).to(device)
                mobile_input = torch.stack([mobile_transform(image) for image in images]).to(device)
                clip_rows.append(normalize(clip_model.encode_image(clip_input).float()).cpu().numpy())
                mobile_features = mobile.avgpool(mobile.features(mobile_input)).flatten(1)
                mobile_rows.append(normalize(mobile_features).cpu().numpy())
            if fingerprint(source) != signature:
                raise ValueError(f"{annotation}: source changed during extraction. Run prepare again.")

            path = cache_path(source, cache_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as cache:
                temporary = Path(cache.name)
                try:
                    np.savez_compressed(
                        cache, fingerprint=signature, frame_indices=[frame.index for frame in frames],
                        clip_question=clip_question[0].cpu().numpy(),
                        clip_images=np.concatenate(clip_rows), policy_images=np.concatenate(mobile_rows),
                    )
                    cache.close()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            print(f"Cached {len(frames)} frames in {path}", flush=True)
