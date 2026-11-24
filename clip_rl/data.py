"""HoloAssist annotation/CSV input and feature-cache validation."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path

from .environment import Clip

CACHE_VERSION = 1


@dataclass(frozen=True)
class Source:
    annotation: Path
    video: Path
    features: Path
    clip_id: str
    session_id: str
    question: str
    start: float
    end: float


@dataclass(frozen=True)
class Frame:
    index: int
    video_time: float
    elapsed: float
    sharpness: float
    gyro: float
    imu_available: bool


def read_source(annotation: Path) -> Source:
    metadata = json.loads(annotation.read_text())
    root = annotation.parent.parent
    source = Source(
        annotation, root / metadata["video_path"], root / metadata["frame_features_path"],
        metadata["clip_id"], metadata["session_id"], metadata["question"],
        float(metadata["clip_start_sec"]), float(metadata["clip_end_sec"]),
    )
    if not source.question.strip() or not source.session_id or not source.clip_id:
        raise ValueError(f"{annotation}: question, session ID and clip ID are required.")
    if not (isfinite(source.start) and isfinite(source.end) and 0 <= source.start < source.end):
        raise ValueError(f"{annotation}: invalid clip bounds.")
    return source


def read_frames(source: Source) -> list[Frame]:
    frames = []
    with source.features.open(newline="") as stream:
        for row in csv.DictReader(stream):
            available = row["imu_available"] == "1"
            frame = Frame(
                int(row["source_frame_index"]), float(row["video_time_sec"]),
                float(row["clip_time_sec"]), float(row["sharpness_raw"]),
                float(row["gyro_rms_past_0_5s"]) if available else 0.0, available,
            )
            if row["imu_available"] not in ("0", "1"):
                raise ValueError(f"{source.features}: invalid IMU availability flag.")
            if not all(map(isfinite, (frame.video_time, frame.elapsed, frame.sharpness, frame.gyro))):
                raise ValueError(f"{source.features}: non-finite feature at frame {frame.index}.")
            if (
                frame.index < 0 or frame.sharpness < 0 or frame.gyro < 0 or frame.elapsed < 0
                or not source.start <= frame.video_time <= source.end
                or abs(frame.elapsed - (frame.video_time - source.start)) > 0.002
            ):
                raise ValueError(f"{source.features}: invalid features at frame {frame.index}.")
            if available:
                imu_time = float(row["imu_time_sec"])
                if not isfinite(imu_time) or imu_time > frame.video_time + 1e-6:
                    raise ValueError(f"{source.features}: future or invalid IMU time at frame {frame.index}.")
            if frames and (frame.index <= frames[-1].index or frame.video_time <= frames[-1].video_time):
                raise ValueError(f"{source.features}: frames must be strictly ordered.")
            frames.append(frame)
    if not frames:
        raise ValueError(f"{source.features}: no frames.")
    return frames


def cache_path(source: Source, directory: Path) -> Path:
    return directory / source.annotation.parent.parent.name / (source.annotation.stem + ".npz")


def fingerprint(source: Source) -> str:
    video_stat = source.video.stat()
    payload = (
        CACHE_VERSION, source.clip_id, source.session_id, source.question, source.start, source.end,
        video_stat.st_size, video_stat.st_mtime_ns,
        hashlib.sha256(source.features.read_bytes()).hexdigest(),
    )
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def question_embedding(question: str) -> tuple[float, ...]:
    """Fixed signed hashing of question words, available before replay."""
    features = [0.0] * 128
    for word in re.findall(r"\w+", question.casefold()):
        digest = hashlib.blake2b(word.encode(), digest_size=2).digest()
        features[digest[0] % len(features)] += 1.0 if digest[1] & 1 else -1.0
    norm = sqrt(sum(value * value for value in features))
    return tuple(value / norm for value in features) if norm else tuple(features)


def load_clip(annotation: Path, directory: Path) -> Clip:
    import numpy as np

    source = read_source(annotation)
    frames = read_frames(source)
    path = cache_path(source, directory)
    with np.load(path, allow_pickle=False) as cache:
        if cache["fingerprint"].item() != fingerprint(source):
            raise ValueError(f"{path}: cache is stale. Run prepare again.")
        if not np.array_equal(cache["frame_indices"], [frame.index for frame in frames]):
            raise ValueError(f"{path}: cached frame indices do not match the CSV.")
        policy_images = cache["policy_images"]
        clip_images = cache["clip_images"]
        clip_question = cache["clip_question"]
    for matrix in (policy_images, clip_images):
        if matrix.ndim != 2 or matrix.shape[0] != len(frames) or matrix.shape[1] == 0:
            raise ValueError(f"{path}: invalid image embedding shape.")
        if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise ValueError(f"{path}: image embeddings must be finite unit vectors.")
    if (
        clip_question.shape != (clip_images.shape[1],) or not np.isfinite(clip_question).all()
        or not np.isclose(np.linalg.norm(clip_question), 1, atol=1e-4)
    ):
        raise ValueError(f"{path}: invalid CLIP question embedding.")
    return Clip(
        source.clip_id, source.session_id, source.question,
        tuple(frame.index for frame in frames), tuple(frame.video_time for frame in frames),
        tuple(frame.elapsed for frame in frames), tuple(frame.sharpness for frame in frames),
        tuple(frame.gyro for frame in frames), tuple(frame.imu_available for frame in frames),
        question_embedding(source.question), policy_images, clip_question, clip_images,
    )
