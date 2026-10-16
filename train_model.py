# Train the Product Review Classification model
# Run: python train_model.py

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Load the preprocessed dataset
df = pd.read_csv("preprocessed_reviews.csv")
df = df.dropna(subset=["cleaned_text", "sentiment"])

X = df["cleaned_text"].astype(str)
y = df["sentiment"].astype(str).str.title()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Logistic Regression and give more attention to less common classes
model = LogisticRegression(max_iter=400, class_weight="balanced")
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

print("Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the model and vectorizer for the Streamlit app
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

# Save a confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=30)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

print("\nSaved model and vectorizer in the models folder.")
