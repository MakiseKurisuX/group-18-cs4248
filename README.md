# CS4248 Group 18 Project - Sarcasm Detection

This repository contains the codebase, datasets, and models for our sarcasm detection project.

## Data

The `/data` directory contains the datasets used for training and evaluating our models:

* **`Sarcasm_Headlines_Dataset.json`**: A news headlines dataset for sarcasm detection, sourced from Kaggle.
  * Source: [News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
* **`Sarcasm_Headlines_Dataset_added.json`**: This dataset extends `Sarcasm_Headlines_Dataset.json` with added data in order to extend our model capabilities to other kinds of headlines.
* **`testing_dataset_final.csv`**: The final test dataset, which includes the original test data along with data added from a few other sources (all additional sources are cited and listed in the project report).

## Models

During the project, we experimented with several transformer models for the sarcasm detection task. In the `models/` directory, we tested the following:

- **BERT**
- **RoBERTa**
- **DeBERTa**

Results of the following models can be seen in our report.

## filter_dataset.py

Filters sarcastic headlines using **Claude Haiku** (`claude-haiku-4-5-20251001`) to enforce the project's scoping decision: keeping only headlines whose sarcasm is detectable from linguistic signals alone, and dropping those that require world knowledge, missing visual context, or cultural background to understand.

Headlines are sent in batches of 100 with structured outputs via `instructor` and Pydantic validation. The script supports checkpointing and exponential-backoff retries, so it can resume safely if interrupted. Non-sarcastic headlines skip the API entirely and are automatically kept.

Applied to all new data entering the pipeline (augmentation and evaluation sets). The original training data is left unfiltered. See Appendix A for the full prompt.

```bash
export ANTHROPIC_API_KEY=your_key_here
python filter_dataset.py
```