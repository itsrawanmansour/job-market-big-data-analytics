# Data — Raw

This folder should contain three files before running the pipeline:

| File | Included in repo? | Notes |
|---|---|---|
| `postings.csv` | ❌ No (too large for GitHub, ~120K rows) | Download from Kaggle (see link below) and place it here |
| `job_skills.csv` | ✅ Yes | Maps `job_id` to skill abbreviations |
| `skills.csv` | ✅ Yes | Maps skill abbreviations to readable skill names |

## Download `postings.csv`

The main dataset file comes from the **LinkedIn Job Postings (2023–2024)** dataset on Kaggle:

https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

1. Download the dataset from the link above (requires a free Kaggle account).
2. Extract it and copy `postings.csv` into this folder (`data/raw/postings.csv`).
3. Once all three files are in place, follow the pipeline steps in the main [README.md](../../README.md).
