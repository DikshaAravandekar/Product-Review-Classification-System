# Product Review Classification System

A student NLP project that classifies product reviews as Positive, Negative, or Neutral.

## Files
- `preprocessed_reviews.csv`: cleaned dataset prepared by Member 1.
- `train_model.py`: trains and evaluates TF-IDF + Logistic Regression.
- `app.py`: Streamlit interface with single-review and CSV classification.
- `requirements.txt`: required libraries.
- `models/`: generated after training.

## Run
```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The updated training script uses TF-IDF unigrams and bigrams and `class_weight="balanced"` to give the less common Neutral and Negative classes more attention. This may improve minority-class predictions, but it does not guarantee every review will be correct. Use the actual evaluation report from your run in the presentation.

CSV upload supports a review-text column named `cleaned_text`, `review`, `Review`, `text`, or `Text`. Results can be downloaded as `classified_reviews.csv`.
