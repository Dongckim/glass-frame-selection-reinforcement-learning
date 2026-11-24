"""Prepare, train and evaluate a causal DQN frame selector."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import mean

from .data import load_clip, read_source


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def positive_float(value):
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number.")
    return number


def config():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Cache OpenCLIP and MobileNet embeddings.")
    train = sub.add_parser("train", help="Train Double DQN on replayed clips.")
    evaluate = sub.add_parser("evaluate", help="Score DQN and fixed-interval sampling.")
    for command in (prepare, train, evaluate):
        command.add_argument("--cache-dir", type=Path, required=True)
        command.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda.")
    for command in (prepare, evaluate):
        command.add_argument("--annotations", type=Path, nargs="+", required=True)
    prepare.add_argument("--batch-size", type=positive_int, default=32)
    train.add_argument("--train", type=Path, nargs="+", required=True)
    train.add_argument("--validation", type=Path, nargs="+", default=[])
    train.add_argument("--budget", type=positive_int, default=3)
    train.add_argument("--sharpness-scale", type=positive_float, default=2000.0)
    train.add_argument("--episodes", type=positive_int, default=100)
    train.add_argument("--eval-every", type=positive_int, default=10)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--batch-size", type=positive_int, default=64)
    train.add_argument("--replay-size", type=positive_int, default=20000)
    train.add_argument("--learning-rate", type=positive_float, default=0.0003)
    train.add_argument("--explore-transmit-prob", type=positive_float, default=0.005)
    for command in (train, evaluate):
        command.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--interval-sec", type=positive_float, default=10.0)
    evaluate.add_argument("--allow-seen-sessions", action="store_true", help="Smoke tests only.")
    return parser


def run_training(args):
    from .dqn import train

    train_sessions = {read_source(path).session_id for path in args.train}
    validation_sessions = {read_source(path).session_id for path in args.validation}
    if overlap := train_sessions & validation_sessions:
        raise ValueError(f"Training and validation sessions overlap: {sorted(overlap)}")
    clips = [load_clip(path, args.cache_dir) for path in args.train]
    validation = [load_clip(path, args.cache_dir) for path in args.validation]
    if len({clip.policy_images.shape[1] for clip in clips + validation}) != 1:
        raise ValueError("Policy embedding widths must agree across clips.")
    train(clips, validation, args)


def run_evaluation(args):
    import torch

    from .dqn import greedy_policy, network
    from .evaluation import evaluate_episode, fixed_interval_policy

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    seen = set(checkpoint["train_sessions"]) | set(checkpoint["validation_sessions"])
    sessions = {read_source(path).session_id for path in args.annotations}
    overlap = seen & sessions
    if overlap and not args.allow_seen_sessions:
        raise ValueError(f"Evaluation overlaps training or validation sessions: {sorted(overlap)}")
    policy = network(checkpoint["width"]).to(args.device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    choose = greedy_policy(policy, args.device)
    budget, scale = checkpoint["budget"], checkpoint["sharpness_scale"]
    rows = {"dqn": [], "fixed_interval": []}
    for path in args.annotations:
        clip = load_clip(path, args.cache_dir)
        dqn = evaluate_episode(clip, budget, scale, choose)
        fixed = evaluate_episode(clip, budget, scale, fixed_interval_policy(args.interval_sec))
        rows["dqn"].append(dqn)
        rows["fixed_interval"].append(fixed)
    print(json.dumps({
        "budget": budget,
        "fixed_interval_sec": args.interval_sec,
        "checkpoint_episode": checkpoint["episode"],
        "seed": checkpoint["seed"],
        "seen_sessions": sorted(overlap),
        "methods": {
            name: {"mean_return": mean(row["return"] for row in scores), "clips": scores}
            for name, scores in rows.items()
        },
    }, indent=2))


def main():
    parser = config()
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from .prepare import prepare

            prepare(args.annotations, args.cache_dir, args.batch_size, args.device)
        elif args.command == "train":
            if args.replay_size < args.batch_size or args.explore_transmit_prob > 1:
                parser.error("Replay size must cover a batch and exploration probability must be at most 1.")
            run_training(args)
        else:
            run_evaluation(args)
    except ModuleNotFoundError as error:
        parser.exit(1, f"{error}. Install clip_rl/requirements.txt in the active environment.\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
