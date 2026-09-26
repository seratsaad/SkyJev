"""Task definitions: one YAML list per category under tasks/."""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("lookup", "search", "multistep", "interactive", "adversarial")


@dataclass
class Task:
    id: str
    category: str
    goal: str
    start_url: str
    verifier: dict
    gold_url: str | None = None
    # A pattern naming the goal page itself, used only to ask "did the agent ever stand on the goal
    # page?" from its recorded URLs. Deliberately separate from the verifier: several verifier URL
    # clauses are loose necessary conditions ("left the home page") meant to pair with text checks,
    # and read alone they accept nearly every page visited. None means the goal is not identifiable
    # from the URL, usually because the site submits its search by POST, so the result page has the
    # same URL whatever was searched for. For those tasks the question is not asked.
    goal_url: str | None = None
    difficulty: str = "unknown"
    timeout_seconds: int = 90
    verify_timeout_seconds: float = 12.0
    expected_limits: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        self.goal = " ".join(self.goal.split())
        if "type" not in self.verifier:
            raise ValueError(f"{self.id}: verifier needs a type")


TASK_SETS = {"dev": "tasks", "heldout": "tasks_heldout"}


def load_tasks(paths: list[str] | None = None, task_set: str = "dev") -> list[Task]:
    """`dev` is the original 15; `heldout` was written before any JEV_PLUS fix and is not tuned on."""
    paths = paths or sorted(glob.glob(str(ROOT / TASK_SETS[task_set] / "*.yaml")))
    tasks: list[Task] = []
    for path in paths:
        for item in yaml.safe_load(Path(path).read_text()) or []:
            tasks.append(Task(**item))
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    return tasks


def by_id(task_id: str) -> Task:
    for task in load_tasks():
        if task.id == task_id:
            return task
    raise KeyError(task_id)
