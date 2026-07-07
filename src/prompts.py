"""Load and access the ethical prompts used in the study."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Prompt:
    id: str
    category: str
    text: str


class PromptLibrary:
    def __init__(self, prompts: List[Prompt]):
        self.prompts = prompts

    @classmethod
    def from_json(cls, path: Path) -> "PromptLibrary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [Prompt(**item) for item in data["prompts"]]
        return cls(prompts)

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)

    def by_category(self, category: str) -> List[Prompt]:
        return [p for p in self.prompts if p.category == category]

    def categories(self) -> List[str]:
        return sorted(set(p.category for p in self.prompts))