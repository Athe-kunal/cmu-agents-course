"""Trajectory -> training data. *** STUDENT IMPLEMENTATION FILE ***

The heart of the data pipeline: turning a multi-turn episode into stitched
token sequences with correct masking, logprob alignment, and advantage
broadcasting. Read the TokenSequence docstring in `scottyrl/types.py`
carefully before starting, especially the off-by-one: position j of
mask/logprobs/advantages describes the *target* token tokens[j+1].

Implement the three functions below. You may add private helpers in this
file, but these signatures must not change.
"""

from __future__ import annotations

from scottyrl.types import TokenSequence, Trajectory


def extension_suffix(accumulated: list[int], prompt_tokens: list[int]) -> list[int] | None:
    """Check the sequence-extension property between an accumulated token
    sequence and the next call's prompt.

    Contract: if `accumulated` is a (non-strict) prefix of `prompt_tokens`,
    return the remaining suffix of `prompt_tokens` (the tokens of the newly
    rendered tail: turn markers, environment observation, generation prompt;
    may be empty). Otherwise return None: the extension is broken and the
    caller must start a new sequence.

    Examples:
        extension_suffix([1, 2, 3], [1, 2, 3, 9, 9]) -> [9, 9]
        extension_suffix([1, 2, 3], [1, 2, 3])       -> []
        extension_suffix([1, 2, 3], [1, 9, 3, 4])    -> None
        extension_suffix([1, 2, 3], [1, 2])          -> None
        extension_suffix([], [1, 2])                 -> [1, 2]  (an empty sequence is a prefix of anything)
    """
    if prompt_tokens[:len(accumulated)] != accumulated:
        return None
    return prompt_tokens[len(accumulated):]


def stitch_trajectory(traj: Trajectory) -> list[TokenSequence]:
    """Stitch an episode's LLM calls into training sequences.

    Output contract:
      - Walk the calls in order, maintaining one growing token sequence of
        context tokens (prompt + rendered env observations) interleaved with
        completion tokens. A call whose prompt extends the sequence built so
        far continues it: only the new context suffix is appended before the
        call's completion. A call that breaks the extension property ends the
        current sequence and starts a new one from that call's full prompt.
      - In every emitted TokenSequence: mask is 1 at exactly the target
        positions whose token came from a completion; logprobs carries the
        sampling logprob at those positions and 0.0 everywhere else;
        advantages is all zeros at this stage.
      - Never emit a sequence with fewer than 2 tokens or with no action
        (mask == 1) positions; there would be nothing to train on.

    Example -- two calls whose prompts extend:
        call 0: prompt=[10, 11], completion=[20, 21], logprobs=[-0.2, -0.3]
        call 1: prompt=[10, 11, 20, 21, 30], completion=[40], logprobs=[-0.4]

    Before call 1, the accumulated sequence is [10, 11, 20, 21], so only
    prompt suffix [30] is appended. The single result is:

        tokens     = [10, 11, 20, 21, 30, 40]
        targets    =      [11, 20, 21, 30, 40]  (= tokens[1:])
        mask       =      [ 0,  1,  1,  0,  1]
        logprobs   =      [0.0, -0.2, -0.3, 0.0, -0.4]
        advantages =      [0.0, 0.0, 0.0, 0.0, 0.0]

    Example -- the second prompt breaks the extension property:
        call 0: prompt=[10, 11], completion=[20, 21]
        call 1: prompt=[10, 11, 30], completion=[40]

    Since [10, 11, 20, 21] is not a prefix of call 1's prompt, this emits two
    results. Their tokens and masks are:

        tokens=[10, 11, 20, 21], mask=[0, 1, 1]
        tokens=[10, 11, 30, 40], mask=[0, 0, 1]

    Returns: list of TokenSequence (possibly empty).
    """
    sequences: list[TokenSequence] = []
    tokens: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []

    for call in traj.calls:
        suffix = extension_suffix(tokens, call.prompt_tokens) if tokens else None
        if suffix is None and len(tokens) >= 2 and 1 in mask:
            sequences.append(TokenSequence(
                list(tokens), list(mask), list(logprobs), [0.0] * len(mask),
            ))
        if suffix is None:
            prompt = call.prompt_tokens
            action_logprobs = call.completion_logprobs if prompt else call.completion_logprobs[1:]
            context_zeros = max(len(prompt) - 1, 0)
            tokens = list(prompt) + list(call.completion_tokens)
            mask = [0] * context_zeros + [1] * len(action_logprobs)
            logprobs = [0.0] * context_zeros + [float(lp) for lp in action_logprobs]
            continue
        tokens.extend(suffix + call.completion_tokens)
        mask.extend([0] * len(suffix) + [1] * len(call.completion_tokens))
        logprobs.extend([0.0] * len(suffix) + [float(lp) for lp in call.completion_logprobs])

    if len(tokens) >= 2 and 1 in mask:
        sequences.append(TokenSequence(
            list(tokens), list(mask), list(logprobs), [0.0] * len(mask),
        ))
    return sequences

def trajectory_to_training_sequences(traj: Trajectory, advantage: float) -> list[TokenSequence]:
    """Stitch a trajectory and broadcast a scalar advantage over its action
    tokens: every position with mask == 1 gets `advantage`, every other
    position stays 0.0. (In REINFORCE/GRPO, all tokens the policy generated
    in an episode share the episode-level advantage.)
    """
    sequences = stitch_trajectory(traj)
    for sequence in sequences:
        # all the tokens get the same advantage
        sequence.advantages = [
            float(advantage) if bit == 1 else 0.0 for bit in sequence.mask
        ]
    return sequences
