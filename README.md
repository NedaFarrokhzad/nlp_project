# Ethical Reasoning in Large Language Models: A Comparative Study of Gemini Flash vs. Gemini Flash-Lite

A computational study comparing how two variants of Google's Gemini language model family respond to morally ambiguous prompts.

**Course:** Natural Language Processing
**Institution:** Universita degli Studi di Milano
**Instructor:** Prof. Alfio Ferrara
**Academic Year:** 2025-2026
**Author:** Neda Farrokhzad

---

## Research Question

When the same morally ambiguous prompt is given to two models from the same family, do they respond in systematically different ways, and what linguistic, strategic, and emotional patterns characterise those differences?

---

## Methodology

The study compares Gemini 2.5 Flash and Gemini 2.5 Flash-Lite across 30 ethical prompts divided into six categories: trolley dilemmas, medical ethics, privacy vs. safety, honesty vs. kindness, free speech vs. harm, and resource allocation.

Each prompt is sent to both models, producing 60 responses. Each response is analysed with two complementary classifiers:

- Keyword-based classifier counts hedge words, refusal phrases, direct commitments, and balanced framings, then assigns a categorical strategy label using a priority rule. Keyword counts are also normalised per 100 words to control for response length differences.
- Transformer-based sentiment classifier uses the cardiffnlp/twitter-roberta-base-sentiment-latest RoBERTa model from Hugging Face to assign positive, negative, and neutral probability scores plus a discrete label.

---

## Installation

1. Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate

2. Install dependencies:

    pip install -r requirements.txt

3. Create a .env file in the project root:

    GOOGLE_API_KEY=your_key_here

Never commit your .env file to version control. It contains your API key. The .gitignore already excludes it.

---

## How to Run

Run the keyword pipeline:

    python run_keyword_analysis.py

Run the sentiment pipeline:

    python run_sentiment_analysis.py

Open the notebook demo:

    jupyter notebook notebooks/demo.ipynb

The experiment already ran; results are in data/responses.csv. To re-collect (takes several days due to rate limits):

    python run_experiment.py

---

## Repository Structure

    nlp_project/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── run_experiment.py
    ├── run_keyword_analysis.py
    ├── run_sentiment_analysis.py
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── prompts.py
    │   ├── llm_clients.py
    │   ├── experiment.py
    │   ├── keyword_analysis.py
    │   ├── keyword_visualization.py
    │   ├── sentiment_analysis.py
    │   └── sentiment_visualization.py
    ├── data/
    │   ├── prompts.json
    │   ├── responses.csv
    │   ├── keyword_results.csv
    │   ├── sentiment_results.csv
    │   └── case_studies.txt
    ├── results/
    │   ├── 01_strategy_distribution.png
    │   ├── 02_hedge_by_category.png
    │   └── 03_sentiment_by_model.png
    ├── notebooks/
    │   └── demo.ipynb
    └── report.pdf

---

## Design Notes

Object-oriented architecture. LLMClient is an abstract base class defining the contract every model wrapper must satisfy. Concrete clients (GeminiClient) inherit from it. This design follows the Template Method pattern and the open-closed principle: adding a new provider requires a new subclass, with zero changes to existing code.

Two-classifier design. KeywordAnalyzer provides transparent, rule-based classification. SentimentAnalyzer uses a pre-trained transformer model. Each has a matching visualizer class in a separate file.

Crash-safe data collection. ExperimentRunner writes every response to disk immediately and skips already-completed pairs on restart.

Separation of concerns. Each module has a single responsibility.