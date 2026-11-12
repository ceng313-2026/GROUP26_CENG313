# GROUP26_CENG313

CENG313 Data Science Course Project — WSDM Cup Multilingual Chatbot Arena.

## Team

- Group ID: 26
- Member 1: Muhammet Kağan Kürkçü — 23118080061
- Member 2: Ömer Coşar — 23118080069

## Competition

[WSDM Cup - Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)

The competition is closed. The provided test file contains three records marked `scored=False`; therefore, no official Kaggle leaderboard score is reported for this submission.

## Repository structure

```text
GROUP26_CENG313/
├── README.md
├── requirements.txt
├── .gitignore
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── utils.py
├── data/          # Local competition data; do not commit data files
└── submissions/   # Submission files or score evidence, when applicable
```

## Data setup

1. Open the competition page and download the competition data after obtaining access to its rules.
2. Create a local `data/` directory in the repository.
3. Place the files in that directory using these names:
   - `train.parquet`
   - `test.parquet`
   - `sample_submission.csv`
4. Do not commit the competition data files to GitHub. The `.gitignore` file excludes local files under `data/`.

## Environment setup

Use Python 3.10 or newer, then install the required packages:

```bash
pip install -r requirements.txt
```

## How to run

1. Start Jupyter Notebook or open the notebooks in Google Colab.
2. Run `assignment1_eda_baseline.ipynb` from top to bottom.
3. The notebook contains the exploratory data analysis, preprocessing, validation, baseline comparisons, and final test prediction workflow.
4. The helper modules are located in `src/`.

For local execution, ensure that the data files are available in the `data/` directory. In Google Colab, update the data directory path to the location where the files are stored in Google Drive.

## Assignment 1 results

The best evaluated model was an enhanced pairwise TF-IDF Logistic Regression model with response-length features.

- Validation accuracy: **60.05%**
- Validation macro F1-score: **0.6001**
- Dummy baseline accuracy: **51.68%**
- Official Kaggle leaderboard score: **Not available — competition closed and test records are unscored.**

The reported validation metrics come from a single 80/20 stratified group split by prompt. Identical prompts were kept within the same partition to reduce prompt leakage. This is a held-out validation result, not an official competition score.

## Notes

- Training and validation data are used for model evaluation; the three unscored test rows are used only to create the required prediction file.
- Long-text outliers were retained because they may represent valid prompts or responses.
- Model preference rates are descriptive and should not be interpreted as absolute model quality.
