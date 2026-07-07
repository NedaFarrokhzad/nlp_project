"""Charts for the keyword analysis results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


STRATEGY_PALETTE = {
    "refusal": "#C0504D",
    "hedged": "#D4A017",
    "balanced": "#4682B4",
    "direct": "#5B8C5A",
    "mixed": "#808080",
}

MODEL_PALETTE = {
    "flash": "#5B4E7A",
    "flash-lite": "#4A7A82",
}


class KeywordVisualizer:
    def __init__(self, analysis_df: pd.DataFrame, output_dir: Path):
        self.df = analysis_df.copy()
        self.df["model_short"] = self.df["model_name"].str.replace("gemini-2.5-", "")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filename}")

    def plot_strategy_distribution(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 6))
        strategy_order = ["balanced", "hedged", "refusal", "direct", "mixed"]
        sns.countplot(
            data=self.df,
            x="model_short",
            hue="strategy",
            order=["flash", "flash-lite"],
            hue_order=strategy_order,
            palette=STRATEGY_PALETTE,
            ax=ax,
        )
        ax.set_title("Response Strategy Distribution by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Number of Responses")
        ax.legend(title="Strategy")
        self._save(fig, "01_strategy_distribution.png")

    def plot_hedge_by_category(self) -> None:
        fig, ax = plt.subplots(figsize=(11, 6))
        means = (
            self.df.groupby(["category", "model_short"])["hedge_per_100_words"]
            .mean()
            .reset_index()
        )
        sns.barplot(
            data=means,
            x="category",
            y="hedge_per_100_words",
            hue="model_short",
            hue_order=["flash", "flash-lite"],
            palette=MODEL_PALETTE,
            ax=ax,
        )
        ax.set_title("Mean Hedge Words per 100 Words, by Category and Model")
        ax.set_xlabel("Ethical Category")
        ax.set_ylabel("Mean Hedge Words per 100 Words")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.legend(title="Model")
        self._save(fig, "02_hedge_by_category.png")

    def make_all(self) -> None:
        self.plot_strategy_distribution()
        self.plot_hedge_by_category()


def run_visualization(
    analysis_path: str = "data/keyword_results.csv",
    output_dir: str = "results",
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    analysis_full = project_root / analysis_path
    output_full = project_root / output_dir

    df = pd.read_csv(analysis_full)
    viz = KeywordVisualizer(df, output_full)
    viz.make_all()
    print(f"All figures saved to {output_full}")


if __name__ == "__main__":
    run_visualization()