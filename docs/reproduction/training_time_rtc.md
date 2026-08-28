# Training-Time RTC

## Purpose

The controller executes actions at 20 Hz while a remote policy server generates the next `[50,28]` chunk. The robot continues consuming the old chunk during inference, so several positions at the front of the new chunk have already elapsed when the response arrives. RTC conditions the new sample on committed actions from the old chunk as a hard prefix, preserving continuity under non-zero latency.

For action horizon `H=50`, inference delay `d` control steps, and a desired executable postfix of at least `s` steps:

```text
d <= H - s
```

The current training configuration uses `s=25` and samples `d` in `[0,25]`. A real deployment should derive this distribution from measured logs rather than assuming 25 is universally optimal.

## Training

The OpenPI PyTorch path uses

```text
x_t = t * noise + (1 - t) * action
```

so `t=0` is the clean-action endpoint. Each batch item samples a prefix length:

- the prefix receives clean expert actions and token timestep zero;
- the postfix receives ordinary Flow Matching noise and timesteps;
- Flow loss excludes the prefix, episode padding, and four padded model dimensions.

The core implementation is in `src/openpi/models_pytorch/pi0_pytorch.py` and `src/openpi/models/pi0_config.py`.

## Deployment

The asynchronous client sends

```python
observation["rtc_prefix"] = {
    "action_prefix": action_prefix,
    "delay": estimated_delay_steps,
}
```

Every denoising step overwrites the committed prefix, and the implementation restores it after the final Euler update. When the new chunk arrives, the client uses the actual elapsed control steps to discard expired entries and queues only the remaining postfix. `RollingChunkController` and `RtcChunker` track request identifiers, execution positions, and recent delays; they reject stale, mismatched, or non-finite results.

## Validation metrics

Offline replay should report inference delay in control steps, first- and second-order differences at chunk seams, replanning rate, and queue exhaustion. Real-robot logs should additionally record joint-limit triggers, emergency stops, tracking error, and lower-body controller status.

```bash
uv run python examples/unitree_g1/eval/simulate_rtc_replay.py
uv run pytest examples/unitree_g1/rtc/rtc_chunker_test.py -q
```

The repository validates scheduler and hard-prefix code paths. It does not establish success on a specific physical G1 task.
