
# Product Review Classification System
# Run: streamlit run app.py

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Product Review Classifier",
    page_icon="🛍️",
    layout="wide"
)

# Page styling
st.markdown("""
<style>
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
}
.subtitle {
    color: #a7b0c0;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🛍️ Product Review Classifier")
st.markdown(
    '<p class="subtitle">Analyze customer feedback using NLP and machine learning.</p>',
    unsafe_allow_html=True
)

# Load the trained model
if not os.path.exists("models/model.pkl") or not os.path.exists("models/vectorizer.pkl"):
    st.error("Model files are missing. Run python train_model.py first.")
    st.stop()


@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()


def predict_review(review):
    features = vectorizer.transform([review])
    prediction = model.predict(features)[0]
    scores = model.predict_proba(features)[0]
    return prediction, scores


# Separate sections
single_tab, csv_tab = st.tabs([
    "✍️ Single Review",
    "📄 Batch CSV Classification"
])


# ---------------------------------
# TAB 1: SINGLE REVIEW
# ---------------------------------
with single_tab:
    st.subheader("Analyze one product review")
    st.write("Enter a review to predict its sentiment.")

    review = st.text_area(
        "Product review",
        height=150,
        placeholder="Example: The watch looks nice, but the battery is poor."
    )

    if st.button("Predict Sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            result, scores = predict_review(review)

            if result == "Positive":
                st.success("😊 Positive Sentiment")
            elif result == "Negative":
                st.error("😞 Negative Sentiment")
            else:
                st.info("😐 Neutral Sentiment")

            st.subheader("Sentiment Scores")

            score_data = sorted(
                zip(model.classes_, scores),
                key=lambda item: item[1],
                reverse=True
            )

            col1, col2, col3 = st.columns(3)

            score_columns = [col1, col2, col3]

            for col, (label, score) in zip(score_columns, score_data):
                with col:
                    st.metric(label, f"{score * 100:.1f}%")
                    st.progress(float(score))


# ---------------------------------
# TAB 2: BATCH CSV
# ---------------------------------
with csv_tab:
    st.subheader("Classify multiple product reviews")
    st.write(
        "Upload a CSV file to classify many reviews at once "
        "and download the results."
    )

    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=["csv"],
        help="Supported columns: cleaned_text, review, Review, text, Text"
    )

    if uploaded_file is not None:
        st.success(
            f"File selected: {uploaded_file.name} "
            f"({uploaded_file.size / (1024 * 1024):.1f} MB)"
        )

        if st.button("Classify CSV", type="primary"):
            try:
                uploaded_file.seek(0)

                results = []
                text_columns = [
                    "cleaned_text", "review", "Review", "text", "Text"
                ]

                progress = st.progress(0)
                status = st.empty()
                rows_done = 0
                error_message = None

                with st.spinner("Classifying reviews..."):
                    # Read large files in smaller chunks
                    for chunk in pd.read_csv(
                        uploaded_file, chunksize=5000
                    ):
                        text_col = next(
                            (c for c in text_columns if c in chunk.columns),
                            None
                        )

                        if text_col is None:
                            error_message = (
                                "Review text column not found. "
                                "Use cleaned_text, review, or text."
                            )
                            break

                        reviews = chunk[text_col].fillna("").astype(str)
                        features = vectorizer.transform(reviews)

                        chunk["Predicted Sentiment"] = model.predict(features)

                        probabilities = model.predict_proba(features)
                        chunk["Prediction Confidence (%)"] = (
                            probabilities.max(axis=1) * 100
                        ).round(1)

                        results.append(chunk)
                        rows_done += len(chunk)

                        status.write(
                            f"Processed {rows_done:,} reviews..."
                        )
                        progress.progress(
                            min(0.99, rows_done / (rows_done + 5000))
                        )

                if error_message:
                    st.error(error_message)

                elif results:
                    output = pd.concat(results, ignore_index=True)

                    progress.progress(1.0)
                    status.empty()

                    st.success(
                        f"Successfully classified {len(output):,} reviews."
                    )

                    # Sentiment summary
                    st.subheader("Sentiment Summary")

                    counts = output["Predicted Sentiment"].value_counts()

                    total = len(output)

                    col1, col2, col3 = st.columns(3)

                    for col, label in zip(
                        [col1, col2, col3],
                        ["Positive", "Negative", "Neutral"]
                    ):
                        count = int(counts.get(label, 0))
                        percentage = count / total * 100 if total else 0

                        with col:
                            st.metric(label, f"{percentage:.1f}%")
                            st.caption(f"{count:,} reviews")

                    # Wide results table
                    st.subheader("Classification Results")
                    st.caption("Showing the first 100 rows.")

                    st.dataframe(
                        output.head(100),
                        use_container_width=True,
                        height=420
                    )

                    st.download_button(
                        "⬇️ Download Classified CSV",
                        data=output.to_csv(index=False).encode("utf-8"),
                        file_name="classified_reviews.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Could not process the CSV: {e}")


st.divider()
