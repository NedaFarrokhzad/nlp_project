from pathlib import Path

from src.prompts import PromptLibrary
from src.llm_clients import GeminiClient
from src.experiment import ExperimentRunner


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    prompts_path = project_root / "data" / "prompts.json"
    output_path = project_root / "data" / "responses.csv"

    prompts = PromptLibrary.from_json(prompts_path)
    print(f"Loaded {len(prompts)} prompts across {len(prompts.categories())} categories")

    clients = [
        GeminiClient(model_name="gemini-2.5-flash"),
        GeminiClient(model_name="gemini-2.5-flash-lite"),
    ]

    runner = ExperimentRunner(
        prompts=prompts,
        clients=clients,
        output_path=output_path,
        delay_seconds=4.0,
    )
    runner.run()