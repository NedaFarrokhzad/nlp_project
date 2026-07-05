import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

from src.llm_clients import LLMClient, LLMResponse
from src.prompts import Prompt, PromptLibrary


class ExperimentRunner:
    def __init__(
        self,
        prompts: PromptLibrary,
        clients: List[LLMClient],
        output_path: Path,
        delay_seconds: float = 4.0,
    ):
        self.prompts = prompts
        self.clients = clients
        self.output_path = output_path
        self.delay_seconds = delay_seconds
        self.completed_keys = self._load_completed_keys()

    def _load_completed_keys(self) -> set:
        if not self.output_path.exists():
            return set()
        try:
            df = pd.read_csv(self.output_path)
            successful = df[df["success"] == True]
            return set(
                zip(successful["prompt_id"], successful["model_name"])
            )
        except Exception:
            return set()

    def _save_response(self, response: LLMResponse, category: str) -> None:
        record = asdict(response)
        record["category"] = category
        df = pd.DataFrame([record])
        write_header = not self.output_path.exists()
        df.to_csv(
            self.output_path,
            mode="a",
            header=write_header,
            index=False,
        )

    def run(self) -> None:
        total_pairs = len(self.prompts) * len(self.clients)
        done = 0

        for prompt in self.prompts:
            for client in self.clients:
                key = (prompt.id, client.model_name)
                done += 1

                if key in self.completed_keys:
                    print(f"[{done}/{total_pairs}] SKIP {prompt.id} / {client.model_name} (already done)")
                    continue

                print(f"[{done}/{total_pairs}] CALL {prompt.id} / {client.model_name}")
                response = client.query(prompt.id, prompt.text)
                self._save_response(response, prompt.category)

                if response.success:
                    self.completed_keys.add(key)
                    print(f"   OK ({response.latency_seconds:.1f}s, {len(response.response_text)} chars)")
                else:
                    print(f"   FAIL: {response.error_message}")

                time.sleep(self.delay_seconds)

        print(f"\nDone. Output: {self.output_path}")