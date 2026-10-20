"""The validator you write. This is the only file you need to modify.

Each function evaluates a single run object and returns the errors it finds.
An empty list means that no errors were found. `validator/runner.py` defines
what a run object contains, and `ASSIGNMENT.md` defines the four error families you may report.
"""
import base64
import io
import re
import warnings

from openai import BadRequestError
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from validator.model import complete, detokenize, tokenize
from validator.prediction import Error, ErrorFamily
from validator.runner import Run


CONTEXT_FRACTION = 0.5
MAX_OUTPUT_TOKENS = 2048
COMPLETION_MAX_TOKENS = 8192
OMISSION = "\n[... omitted to fit model context ...]\n"
COMPACTION_NOTE = "\nSome evidence may be omitted. Do not treat omissions as agent errors.\n"
#: Shrink the image's longest side by this factor each time text compaction alone
#: still overflows. The server's tokenizer does not count image tokens, so we let
#: its context errors drive downscaling instead of estimating a pixel budget.
IMAGE_DOWNSCALE_FACTOR = 0.7
#: Stop shrinking once the longest side reaches this floor; below it the figure is
#: unreadable and further retries cannot help, so we surface the overflow instead.
MIN_IMAGE_LONG_SIDE = 64


def _excerpt(text: str, tokens: list[int], limit: int) -> str:
    """Retain the beginning and end, including the run's final actions."""
    if len(tokens) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return (
        (detokenize(tokens[:head]) if head else "")
        + OMISSION
        + (detokenize(tokens[-tail:]) if tail else "")
    )


def _compact(trajectory: str, inputs: str, fixed_prompt: str) -> tuple[str, str]:
    trajectory_info = tokenize(trajectory)
    context = trajectory_info.get("max_model_len")
    if type(context) is not int or context <= 0:
        raise RuntimeError("Validator server did not report a valid max_model_len")
    trajectory_tokens = trajectory_info["tokens"]
    input_tokens = tokenize(inputs)["tokens"]
    # Reserve half the context for the image, chat template, and tokenization
    # boundary differences. Count fixed text and omission markers separately.
    fixed_tokens = tokenize(fixed_prompt + COMPACTION_NOTE + OMISSION * 2)["tokens"]
    budget = int(context * CONTEXT_FRACTION) - len(fixed_tokens) - MAX_OUTPUT_TOKENS
    if budget < 0:
        raise RuntimeError("Task instructions exceed the baseline's context budget")
    total = len(trajectory_tokens) + len(input_tokens)
    trajectory_budget = budget * len(trajectory_tokens) // max(total, 1)
    return (
        _excerpt(trajectory, trajectory_tokens, trajectory_budget),
        _excerpt(inputs, input_tokens, budget - trajectory_budget),
    )


def _downscale(data: bytes) -> bytes | None:
    """Shrink a PNG's longest side by IMAGE_DOWNSCALE_FACTOR.

    Returns the smaller PNG, or None when the image cannot be shrunk further
    (already at the floor) or cannot be decoded at all.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    longest = max(width, height)
    if longest <= MIN_IMAGE_LONG_SIDE:
        return None
    scale = max(MIN_IMAGE_LONG_SIDE / longest, IMAGE_DOWNSCALE_FACTOR)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if new_size == (width, height):
        return None
    with Image.open(io.BytesIO(data)) as image:
        if mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        resized = image.resize(new_size, Image.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue()


def _context_overflow(error: BadRequestError) -> bool:
    message = str(error).lower()
    return any(phrase in message for phrase in (
        "context_length_exceeded",
        "maximum context length",
        "maximum model length",
        "maximum input length",
    ))


class ReportedError(BaseModel):
    """The comparison is written in evidence, then family names the category."""

    model_config = ConfigDict(extra="forbid")

    evidence: str = Field(min_length=1, max_length=400)
    family: ErrorFamily


class ErrorReport(BaseModel):
    """Structured completion. The only output is the error list."""

    model_config = ConfigDict(extra="forbid")

    errors: list[ReportedError]


_JUDGE_PROMPT = """
Judge this data-visualization run.
Do not use a thinking channel.
For each error, write the comparison in evidence, then set family to the matching category.
Keep each evidence string to one or two sentences.
Return every error that applies. A run may have several.
Return an empty errors list when the run is acceptable.
Count a family only when one of its cases below is present.

execution_failure: the agent failed to produce a valid figure. This family appears alone.

wrong_data: what is drawn is not the data the instruction asked for.
Wrong values: plotted numbers, positions, or derived quantities differ from the requested data or formula, including wrong aggregation, scaling, or normalisation, a function or field sampled over the wrong range or grid, or a bubble size, colour value, or error bar that should encode the data but does not.
Missing data: a requested series, category, group, panel's data, or data point is not drawn, or a dimension is dropped.
Wrong selection: the wrong subset, column, or filter was used, or series that were not requested are mixed in.
Wrong order: points, bars, categories, or sequences differ from the requested order, including unsorted when sorting was asked for, a reversed or shuffled sequence, or panels that show the wrong data.

wrong_chart: the data is right but a requested chart-design requirement is not followed.
Chart type and projection: wrong chart type, grouped bars for stacked, vertical bars for horizontal, 3D requested but drawn in 2D, polar projection missing, contour or surface missing, twin axes missing or used when forbidden, or a zoomed inset or its connector lines missing.
Layout: wrong number or arrangement of subplots, wrong figure size or dpi, or panels that do not share the axes they were asked to share.
Axes: missing or wrong x, y, or z axis label; wrong exact title or label text; wrong axis limits, including not starting from zero, the wrong range, or limits that cut off requested data; wrong axis scale, including linear for log or non-log contour levels; ticks or tick labels shown when asked to be hidden; wrong tick labels or tick-label rotation; spines visible when asked to be hidden; grid missing.
Legend, colour bar, and annotations: legend missing; colour bar missing or not matching the data; a requested annotation or marked point missing or placed at the wrong location; a requested function label or bar value label missing; a requested formula not rendered; the wrong region shaded.
Style: wrong or forbidden colormap, including a non-uniform one when a perceptually uniform one was requested; wrong line style; wrong marker shape; wrong colour for a named series; areas not filled; whiskers, median markers, connector lines, or boxes missing; polygons not closed; requested line offsets not applied.
A required title or axis label that falls entirely outside the saved image is missing, so it is wrong_chart.

hard_to_read: the figure is drawn as requested, but a reader cannot read it, or can read it only with difficulty.
Clipped text: a title, axis label, tick label, or annotation is partly cut off by the edge of the image. A label cut in half is hard_to_read. One entirely outside the image is wrong_chart.
Overlapping text: tick labels, annotations, titles, or labels overlap one another, or markers are drawn on top of the text that labels them.
Elements covering content: a legend, annotation, or text box hides data, labels, or other text. A legend sitting over empty space is fine.
Low contrast or invisible content: text or value labels with contrast below about 2.5:1 against their background, or a series drawn in the background colour, with zero width, or otherwise invisible.
Indistinguishable data: two series or groups that must be told apart are drawn with the same, or nearly the same, colour, line style, and marker, or a colormap merges values that should look different.
Squeezed layout: data, boxes, or panels are crushed into an unreadable sliver by axis limits or by axes that overlap each other.
Also report hard_to_read when labels run together, pale text sits on a light background, a series disappears into the background, a legend covers the data, or series that must be told apart share a color and line style. Report it in addition to wrong_data or wrong_chart. Do not file a readability problem only as one of those.
"""

_REPORTS: dict[str, list[Error]] = {}
_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _input_text(run: Run) -> str:
    return "\n".join(
        f"{name}:\n{path.read_text(errors='replace')}" for name, path in run.inputs.items()
    )


def _prompt(instructions: str, trajectory: str, inputs: str) -> str:
    shown_inputs = inputs or "(none)"
    return (
        f"{_JUDGE_PROMPT}\n"
        f"Task:\n{instructions}\n\n"
        f"Trajectory:\n{trajectory}\n\n"
        f"Input files:\n{shown_inputs}\n"
    )


def _messages(prompt: str, image: bytes | None) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image is None:
        return [{"role": "user", "content": content}]
    encoded = base64.b64encode(image).decode()
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
    )
    return [{"role": "user", "content": content}]


def _response_format() -> dict:
    schema = ErrorReport.model_json_schema()
    schema.pop("$schema", None)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ErrorReport",
            "strict": True,
            "schema": schema,
        },
    }


def _complete_structured(prompt: str, image: bytes | None) -> dict:
    return complete(
        _messages(prompt, image),
        max_tokens=COMPLETION_MAX_TOKENS,
        response_format=_response_format(),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def _cut_off(error: ValidationError) -> bool:
    return any(item["type"] == "json_invalid" for item in error.errors())


def _one_per_family(errors: list[Error]) -> list[Error]:
    """Keep the first evidence for each family. The scorer rejects duplicates."""
    chosen: dict[ErrorFamily, Error] = {}
    for error in errors:
        if error.family in chosen:
            continue
        chosen[error.family] = error
    return [chosen[family] for family in ErrorFamily if family in chosen]


def _completion_text(response: dict) -> str:
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    return _THINKING.sub("", content).strip()


def _parse_report(response: dict) -> list[Error]:
    content = _completion_text(response)
    if not content:
        raise RuntimeError("Validator model returned an empty structured completion")
    report = ErrorReport.model_validate_json(content)
    graded = [
        Error(family=item.family, evidence=item.evidence) for item in report.errors
    ]
    return _one_per_family(graded)


def _complete_shrinking(run_id: str, prompt: str, image: bytes | None) -> dict:
    # Compacted text still overflows only when the image dominates the context.
    # Downscale it in place, retrying until the server accepts it or the figure
    # reaches a floor below which it would be unreadable anyway.
    while True:
        try:
            return _complete_structured(prompt, image)
        except BadRequestError as error:
            if not _context_overflow(error):
                raise
            smaller = _downscale(image) if image is not None else None
            if smaller is None:
                raise
            with Image.open(io.BytesIO(smaller)) as shrunk:
                new_size = shrunk.size
            warnings.warn(
                f"Run {run_id}: still over context; "
                f"downscaling figure to {new_size[0]}x{new_size[1]} and retrying.",
                stacklevel=2,
            )
            image = smaller


def _reported_errors(run: Run) -> list[Error]:
    cached = _REPORTS.get(run.run_id)
    if cached is not None:
        return cached
    inputs = _input_text(run)
    trajectory = str(run.messages)
    image = run.figure.read_bytes() if run.figure else None
    prompt = _prompt(run.instructions, trajectory, inputs)
    try:
        response = _complete_structured(prompt, image)
    except BadRequestError as error:
        if not _context_overflow(error):
            raise
        warnings.warn(
            f"Run {run.run_id}: model context exceeded; "
            f"retrying with text capped at {CONTEXT_FRACTION:.0%} of server context.",
            stacklevel=2,
        )
        shown_trajectory, shown_inputs = _compact(
            trajectory, inputs, _prompt(run.instructions, "", "")
        )
        compacted = _prompt(run.instructions, shown_trajectory, shown_inputs) + COMPACTION_NOTE
        response = _complete_shrinking(run.run_id, compacted, image)
    try:
        report = _parse_report(response)
    except ValidationError as error:
        if not _cut_off(error):
            raise
        warnings.warn(
            f"Run {run.run_id}: structured completion was cut off; retrying with a shorter answer.",
            stacklevel=2,
        )
        shorter = prompt + "\nKeep every evidence string to one sentence.\n"
        report = _parse_report(_complete_structured(shorter, image))
    _REPORTS[run.run_id] = report
    return report


def _errors_for(run: Run, family: ErrorFamily) -> list[Error]:
    return [error for error in _reported_errors(run) if error.family == family]


def judge_execution(run: Run) -> list[Error]:
    if run.figure:
        return _errors_for(run, ErrorFamily.EXECUTION_FAILURE)
    return [
        Error(
            family=ErrorFamily.EXECUTION_FAILURE,
            evidence=f"No figure.png was saved for run {run.run_id}.",
        )
    ]


def judge_data_and_chart(run: Run) -> list[Error]:
    return _errors_for(run, ErrorFamily.WRONG_DATA) + _errors_for(run, ErrorFamily.WRONG_CHART)


def judge_readability(run: Run) -> list[Error]:
    return _errors_for(run, ErrorFamily.HARD_TO_READ)


def validate(run: Run) -> list[Error]:
    return judge_execution(run) + judge_data_and_chart(run) + judge_readability(run)


