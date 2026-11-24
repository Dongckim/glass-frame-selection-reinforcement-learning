"""Causal frame selection and server-side CLIP reward shared by policies."""

from dataclasses import dataclass
from math import isfinite, sumprod
from typing import Sequence

WAIT = 0
TRANSMIT = 1


@dataclass(frozen=True)
class Clip:
    clip_id: str
    session_id: str
    question: str
    frame_indices: tuple[int, ...]
    video_times: tuple[float, ...]
    elapsed_times: tuple[float, ...]
    sharpness: tuple[float, ...]
    gyro: tuple[float, ...]
    imu_available: tuple[bool, ...]
    policy_question: Sequence[float]
    policy_images: Sequence[Sequence[float]]
    clip_question: Sequence[float]
    clip_images: Sequence[Sequence[float]]


def positive_cosine(unit_a: Sequence[float], unit_b: Sequence[float]) -> float:
    return max(0.0, min(1.0, float(sumprod(unit_a, unit_b))))


def scale_sharpness(raw: float, scale: float) -> float:
    return raw / (raw + scale)


class FrameSelection:
    def __init__(self, clip: Clip, budget: int, sharpness_scale: float):
        if budget < 1 or not isfinite(sharpness_scale) or sharpness_scale <= 0:
            raise ValueError("Budget and sharpness scale must be positive.")
        self.clip = clip
        self.budget = budget
        self.remaining = budget
        self.index = 0
        self.selected: list[int] = []
        self.quality = tuple(scale_sharpness(q, sharpness_scale) for q in clip.sharpness)

    @property
    def done(self) -> bool:
        return self.index == len(self.clip.frame_indices)

    def observation(self) -> tuple[float, ...]:
        if self.done:
            raise ValueError("The episode has ended.")
        i = self.index
        clip = self.clip
        elapsed = clip.elapsed_times[i]
        current = clip.policy_images[i]
        since_transmit = elapsed - clip.elapsed_times[self.selected[-1]] if self.selected else 0.0
        similarity = max(
            (positive_cosine(current, clip.policy_images[j]) for j in self.selected),
            default=0.0,
        )
        return (
            *clip.policy_question,
            *current,
            self.quality[i],
            clip.gyro[i],
            float(clip.imu_available[i]),
            elapsed / 30.0,
            self.remaining / self.budget,
            self.budget / 5.0,
            since_transmit / 30.0,
            float(bool(self.selected)),
            similarity,
        )

    def step(self, action: int) -> float:
        if self.done:
            raise ValueError("The episode has ended.")
        if action not in (WAIT, TRANSMIT):
            raise ValueError(f"Invalid action {action}.")
        reward = 0.0
        if action == TRANSMIT:
            if self.remaining == 0:
                raise ValueError("Transmission budget exhausted.")
            i = self.index
            embedding = self.clip.clip_images[i]
            relevance = positive_cosine(embedding, self.clip.clip_question)
            redundancy = max(
                (positive_cosine(embedding, self.clip.clip_images[j]) for j in self.selected),
                default=0.0,
            )
            reward = self.quality[i] * relevance * (1.0 - redundancy) / self.budget
            self.selected.append(i)
            self.remaining -= 1
        self.index += 1
        return reward
