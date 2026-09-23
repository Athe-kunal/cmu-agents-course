# Token usage: context compaction on vs. off

Instance: `django__django-15368` (SWE-bench), model `accounts/fireworks/models/deepseek-v4-flash-0731`,
sandbox backend Daytona, step limit 200. One run per configuration.

## Setup

| Run | Command | Trajectory |
|---|---|---|
| Compaction on | `COMPACT_THRESHOLD=6000 make run-swebench-agent INSTANCE=django__django-15368` | `django__django-15368-trajectory.json` |
| Baseline (off) | `COMPACT_THRESHOLD=0 make run-swebench-agent INSTANCE=django__django-15368` | `django__django-15368-baseline-trajectory.json` |

Compaction summarizes everything before the most recent assistant step (`compaction_keep_recent_steps=1`) into one
summary message once the estimated prompt reaches the threshold. Each compaction is an extra model call.

## Results

| | Compaction (6000) | Baseline (off) |
|---|---|---|
| Steps | 28 | 47 |
| Compaction calls (non-empty summaries) | 14 (13) | 0 |
| Peak prompt tokens | 6,322 | 45,645 |
| Action-step tokens | 125,193 | 1,232,398 |
| Summarizer tokens | 62,698 | 0 |
| **Total tokens** | **187,891** | **1,232,398** |
| Graded with `make check-swebench` | RESOLVED: yes | RESOLVED: yes |

Total tokens with compaction are about 15% of the baseline (about 85% fewer).

## Observations

- **Context stays flat.** With compaction the prompt never exceeds about 6.3k tokens. Without it, the prompt grows
  with every step and reaches 45.6k, and each step re-sends the whole history.
- **The summarizer is not free.** It accounts for 62,698 of the 187,891 tokens (about a third). The saving comes from
  the action steps, which shrank from 1.23M to 125k tokens.
- **Both patches are correct.** Each changes `isinstance(attr, Expression)` to `hasattr(attr, "resolve_expression")`
  in `django/db/models/query.py`. The baseline also removed the now-unused `Expression` import. Both pass the
  FAIL_TO_PASS test and all PASS_TO_PASS tests.
- **Fewer steps with compaction (28 vs 47).** With one run each this may be run-to-run variance, not an effect of
  compaction.

## Summarizer quality

The first compaction runs on the chess task showed the summarizer misbehaving. In `part1-compaction-trajectory.json`
only 6 of 22 compactions produced a usable summary. The rest were empty (reasoning used the 1,200-token budget or the
model tried to call a tool), and four "summaries" were the model continuing the task, such as a bash snippet or the
final report.

The fix was to frame the transcript as data: the system prompt says not to continue the task or call tools, the
transcript is wrapped in `<transcript>` tags, and the final instruction asks for the summary. Results:

| Run | Compactions | Non-empty | Hit token limit |
|---|---|---|---|
| Chess, before framing fix | 22 | 6 | 13 |
| Chess, after framing fix | 20 | 18 | 8 |
| django__django-15368, after framing fix | 14 | 13 | not measured |

## Caveats

- One run per configuration. Steps, tokens and even correctness vary between runs of the same model.
- Token counts come from each response's `usage` field. The compaction summarizer's own usage is in
  `compactions[].compaction_response.usage`.
- Some summaries still end with `finish=length` (truncated at `compaction_max_tokens`, default 1,200). Raising
  `--compaction-max-tokens` would avoid that.
- With a 6,000-token threshold, compaction re-triggers on most steps. A higher threshold would mean fewer summarizer
  calls, but a larger peak prompt.
