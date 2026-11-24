"""Policy-independent scoring and selected-frame export."""

from math import floor

from .environment import FrameSelection, TRANSMIT, WAIT


def evaluate_episode(clip, budget, sharpness_scale, choose):
    env = FrameSelection(clip, budget, sharpness_scale)
    reward = 0.0
    while not env.done:
        if not env.remaining:
            break  # The only remaining action is Wait, with zero reward.
        reward += env.step(choose(env.observation(), clip.elapsed_times[env.index]))
    return {
        "clip_id": clip.clip_id,
        "session_id": clip.session_id,
        "return": reward,
        "selected_frames": [
            {"source_frame_index": clip.frame_indices[i], "video_time_sec": clip.video_times[i]}
            for i in env.selected
        ],
    }


def fixed_interval_policy(interval_sec):
    next_time = 0.0

    def choose(observation, elapsed_sec):
        nonlocal next_time
        if elapsed_sec < next_time:
            return WAIT
        next_time = (floor(elapsed_sec / interval_sec) + 1) * interval_sec
        return TRANSMIT

    return choose
