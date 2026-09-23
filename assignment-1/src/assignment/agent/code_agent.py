"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
    load_skill_directory,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment


def build_system_information(environment: Environment) -> str:
    """Format the required <system_information> block from the environment."""

    payload = {
        "machine": environment.machine,
        "release": environment.release,
        "system": environment.system,
        "version": environment.version,
    }
    return (
        "<system_information>\n"
        f"{json.dumps(payload, indent=2)}\n"
        "</system_information>"
    )


def _inject_skills_system_prompt(skills_path: str) -> str:
    """Format available-skill metadata for the system prompt catalog."""

    if not skills_path:
        return ""
    skills = load_skill_directory(Path(skills_path))
    if not skills:
        return ""
    lines = [
        "Available skills. Call invoke_skill with a skill name to load full instructions."
    ]
    for skill in skills.values():
        lines.append(skill["metadata"])
    return "\n".join(lines)


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        self.system_prompt = (
            "You are a software engineering agent working in a terminal.\n"
            "Inspect the repository, reproduce the issue, and implement a fix "
            "using the available tools. Reason about each observation before "
            "the next action.\n"
            f"{build_system_information(self.env)}"
        )
        self.task_prompt = self.task
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        skill_catalog = _inject_skills_system_prompt(
            str(self.skills_path) if self.skills_path else ""
        )
        if skill_catalog:
            self.system_prompt += "\n" + skill_catalog

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations = []
        for tool_call in tool_calls:
            call_id = tool_call.get("id", "")
            function = tool_call.get("function", {})
            name = function.get("name")
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError as exc:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Failed to parse tool arguments: {exc}",
                    }
                )
                continue

            if name == "execute":
                result = self.env.execute(
                    arguments.get("command"),
                    shell=arguments.get("shell", True),
                    cwd=arguments.get("cwd"),
                    timeout=arguments.get("timeout"),
                    env=arguments.get("env"),
                )
                content = format_tool_output(result)
            elif name == "send_message":
                content = str(arguments.get("summary", ""))
                self.finished = True
            elif name == "invoke_skill":
                skill_name = str(arguments.get("name", ""))
                skill = self.skills.get(skill_name)
                if skill is None:
                    available = ", ".join(self.skills) or "none"
                    content = f"Unknown skill: {skill_name!r}. Available skills: {available}"
                else:
                    content = skill["content"]
            else:
                content = f"Unknown tool: {name}"

            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        return observations
