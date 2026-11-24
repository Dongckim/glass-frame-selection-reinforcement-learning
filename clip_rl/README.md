# CLIP reward and DQN frame selection

Run commands from the parent directory with Python 3.13 and FFmpeg/FFprobe.
OpenCLIP ViT-B/32 and MobileNetV3 Small download pretrained weights the first
time they run. The video and question are encoded locally. Use `--device cuda`
on a GPU server if PyTorch has CUDA support.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r clip_rl/requirements.txt

annotation=/var/tmp/holoassist-share/clip_001/annotations/clip_001_draft.json
work_dir=/tmp/holoassist-dqn
python -m clip_rl prepare --annotations "$annotation" --cache-dir "$work_dir/features"
python -m clip_rl train --train "$annotation" --cache-dir "$work_dir/features" \
  --checkpoint "$work_dir/dqn.pt" --episodes 10
python -m clip_rl evaluate --annotations "$annotation" --cache-dir "$work_dir/features" \
  --checkpoint "$work_dir/dqn.pt" --allow-seen-sessions
```

These commands replay the same clip to check the pipeline. For an experiment,
pass session-disjoint paths to `train --train`, `train --validation`, and
`evaluate --annotations`, and omit `--allow-seen-sessions`. Prepare all listed
annotations first. Evaluation reports mean surrogate return and selected source
frame indices and times for Double DQN and a fixed-interval baseline. The fixed
interval starts at elapsed time zero and uses the same budget as Double DQN.
The selected indices can be used for downstream VLM QA evaluation.

Policy inputs are MobileNet image embeddings, a hashed bag-of-words question
embedding, sharpness, recent gyroscope RMS and its availability flag, elapsed
time, budget, time since the last transmission, and similarity to previously
transmitted MobileNet embeddings. OpenCLIP text and image embeddings are used
inside the environment to score transmissions. The environment uses
`q = raw / (raw + scale)`, with a provisional scale of 2000. Set
`--sharpness-scale` using training data when running an experiment. The policy
and baseline both use the scale and budget recorded in the checkpoint.
