# Dataset Information

## KUC Drug Review Dataset (Kaggle)

Source: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

### Download Instructions

1. Install kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   kaggle datasets download jessicali9530/kuc-hackathon-winter-2018 -p data/raw --unzip
   ```

### Files Expected in `data/raw/`
- `drugsComTrain_raw.tsv` — training split (~161k rows)
- `drugsComTest_raw.tsv` — test split (~53k rows)

### Columns
- `drugName` — name of the drug reviewed
- `condition` — medical condition being treated
- `review` — patient-written text review
- `rating` — patient satisfaction rating (1–10)
- `date` — date of review (format: "Month DD, YYYY")
- `usefulCount` — number of users who found the review helpful
