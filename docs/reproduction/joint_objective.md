# Hierarchical FAST + Flow Objective

## Two training branches

Each batch produces two isolated language sequences:

1. The Flow prefix contains three images, the overall instruction, the current subtask, and discrete proprioception. It never contains future expert actions.
2. The FAST teacher-forcing sequence appends `Subtask:` text and FAST action tokens to the same observation prefix and applies a causal cross-entropy mask.

The continuous Action Expert receives the Flow prefix and a noisy `[50,32]` action suffix and learns a target vector field. Inference uses ten Euler Flow steps by default.

Compressed FAST tokens do not map one-to-one to individual control steps. Samples containing repeat-last tail padding therefore skip FAST cross-entropy, while the continuous Flow branch still uses its per-step mask. This prevents synthetic tail actions from becoming discrete-action supervision.

## Loss and Knowledge Insulation

\[
L_{FAST}=-\sum_i m_i\log p(z_i\mid z_{<i},o)
\]

\[
x_t=t\epsilon+(1-t)a,\qquad v^*=\epsilon-a,\qquad
L_{FM}=\operatorname{MaskedMean}\lVert v_\theta(x_t,t,o)-v^*\rVert^2
\]

The implementation first builds the `L_FAST` graph. It then temporarily disables gradients for PaliGemma parameters while constructing the `L_FM` graph, restores their parameter state, and performs one backward pass over the weighted total loss. This explicit Stop-Gradient boundary allows the VLM to receive FAST cross-entropy gradients while the Flow objective trains the Action Expert and action projections without modifying the VLM.

The joint-objective trainer is intentionally limited to one training process. A DDP reducer needs dedicated integration when two model forwards feed one backward pass. Flow-only training retains DDP support.

This is an engineering reproduction of a public Knowledge Insulation concept, not official π0.7 source code. `tests/test_core_pipeline.py` validates the gradient boundary with a small analytically inspectable network when PyTorch is available.

## Inference

When `plan_subtask` is enabled, PaliGemma first generates one current subtask autoregressively. That text enters the normal Flow prefix, and the Action Expert generates one continuous action chunk. The system plans one current subtask at a time rather than producing a complete subtask list. If no valid `Subtask:` segment can be parsed, the policy falls back to the overall task prompt.
