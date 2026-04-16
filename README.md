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