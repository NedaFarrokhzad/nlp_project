from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


SENTIMENT_PALETTE = {
    "positive": "#5B8C5A",  # muted forest green
    "neutral": "#808080",   # gray
    "negative": "#C0504D",  # muted red-brown
}


class SentimentVisualizer:
    def __init__(self, sentiment_df: pd.DataFrame, output_dir: Path):
        self.df = sentiment_df.copy()
        self.df["model_short"] = self.df["model_name"].str.replace("gemini-2.5-", "")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filename}")

    def plot_sentiment_by_model(self) -> None:
        means = (
            self.df.groupby("model_short")[
                ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]
            ]
            .mean()
            .reset_index()
        )

        long_df = means.melt(
            id_vars="model_short",
            var_name="sentiment",
            value_name="probability",
        )
        long_df["sentiment"] = long_df["sentiment"].str.replace("sentiment_", "")

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(
            data=long_df,
            x="model_short",
            y="probability",
            hue="sentiment",
            order=["flash", "flash-lite"],
            hue_order=["positive", "neutral", "negative"],
            palette=SENTIMENT_PALETTE,
            ax=ax,
        )
        ax.set_title("Mean Sentiment Probability by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean Sentiment Probability")
        ax.legend(title="Sentiment")
        self._save(fig, "03_sentiment_by_model.png")

    def make_all(self) -> None:
        self.plot_sentiment_by_model()


def run_visualization(
    sentiment_path: str = "data/sentiment_results.csv",
    output_dir: str = "results",
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    sentiment_full = project_root / sentiment_path
    output_full = project_root / output_dir

    print(f"Loading sentiment results from {sentiment_full}")
    df = pd.read_csv(sentiment_full)
    viz = SentimentVisualizer(df, output_full)
    viz.make_all()
    print(f"All figures saved to {output_full}")


if __name__ == "__main__":
    run_visualization()