"""Double DQN with uniform replay and a periodically copied target network."""

import random
import tempfile
from collections import deque
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from .environment import FrameSelection, TRANSMIT, WAIT
from .evaluation import evaluate_episode


def network(width):
    return nn.Sequential(
        nn.Linear(width, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


def greedy_policy(model, device):
    def choose(observation, elapsed_sec):
        with torch.inference_mode():
            vector = torch.tensor(observation, dtype=torch.float32, device=device)
            return int(model(vector).argmax().item())

    return choose


def update(policy, target, optimizer, batch, device):
    states, actions, rewards, next_states = zip(*batch)
    tensors = torch.tensor(states, dtype=torch.float32, device=device)
    action_indices = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    predicted = policy(tensors).gather(1, action_indices).squeeze(1)
    with torch.no_grad():
        expected = torch.tensor(rewards, dtype=torch.float32, device=device)
        active = [i for i, state in enumerate(next_states) if state is not None]
        if active:
            following = torch.tensor(
                [next_states[i] for i in active], dtype=torch.float32, device=device,
            )
            next_actions = policy(following).argmax(dim=1, keepdim=True)
            expected[active] += target(following).gather(1, next_actions).squeeze(1)
    loss = nn.functional.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()


def train(clips, validation, args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    width = len(FrameSelection(clips[0], args.budget, args.sharpness_scale).observation())
    policy = network(width).to(device)
    target = network(width).to(device)
    target.load_state_dict(policy.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    replay = deque(maxlen=args.replay_size)
    train_sessions = sorted({clip.session_id for clip in clips})
    validation_sessions = sorted({clip.session_id for clip in validation})
    behavior = greedy_policy(policy, device)
    best = float("-inf")
    steps = updates = 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(1, args.episodes + 1):
        env = FrameSelection(rng.choice(clips), args.budget, args.sharpness_scale)
        epsilon = max(0.05, 1.0 - (episode - 1) / max(1.0, args.episodes * 0.75))
        total = 0.0
        observation = env.observation()
        while True:
            if rng.random() < epsilon:
                action = TRANSMIT if rng.random() < args.explore_transmit_prob else WAIT
            else:
                action = behavior(observation, 0.0)
            reward = env.step(action)
            following = None if env.done or not env.remaining else env.observation()
            replay.append((observation, action, reward, following))
            total += reward
            steps += 1
            if len(replay) >= args.batch_size and steps % 4 == 0:
                update(policy, target, optimizer, rng.sample(replay, args.batch_size), device)
                updates += 1
                if updates % 100 == 0:
                    target.load_state_dict(policy.state_dict())
            if following is None:
                break  # The value of future forced Wait actions is zero.
            observation = following

        if episode % args.eval_every != 0 and episode != args.episodes:
            continue
        score = None
        if validation:
            score = mean(
                evaluate_episode(clip, args.budget, args.sharpness_scale, behavior)["return"]
                for clip in validation
            )
        metrics = f"episode={episode} train_return={total:.6f} updates={updates}"
        if score is not None:
            metrics += f" validation_return={score:.6f}"
        print(metrics, flush=True)
        if (score is not None and score > best) or (score is None and episode == args.episodes):
            if score is not None:
                best = score
            checkpoint = {
                "model": policy.state_dict(),
                "width": width,
                "budget": args.budget,
                "sharpness_scale": args.sharpness_scale,
                "train_sessions": train_sessions,
                "validation_sessions": validation_sessions,
                "episode": episode,
                "seed": args.seed,
            }
            with tempfile.NamedTemporaryFile(dir=args.checkpoint.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    torch.save(checkpoint, stream)
                    stream.close()
                    temporary.replace(args.checkpoint)
                finally:
                    temporary.unlink(missing_ok=True)
