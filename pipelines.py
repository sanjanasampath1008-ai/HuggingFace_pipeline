import os
import pandas as pd

from transformers import pipeline


# ==========================================================
# HUGGING FACE PIPELINES - WEEK 7
# ==========================================================

print("\n" + "=" * 70)
print("       HUGGING FACE PIPELINES - WEEK 7 PROJECT")
print("=" * 70)


# ==========================================================
# 1. CREATE DATASET FOLDER
# ==========================================================

os.makedirs("dataset", exist_ok=True)


# ==========================================================
# 2. LOAD INPUT DATA
# ==========================================================

input_file = "dataset/input_text.csv"

df = pd.read_csv(input_file)

print("\nInput data loaded successfully!")

print("\nInput Dataset:")
print(df)


# ==========================================================
# 3. SENTIMENT ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("1. SENTIMENT ANALYSIS")
print("=" * 70)


sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


sentiment_data = []

sentiment_texts = df[df["task"] == "Sentiment"]


for _, row in sentiment_texts.iterrows():

    result = sentiment_pipeline(row["text"])[0]

    sentiment_data.append({
        "id": row["id"],
        "text": row["text"],
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    })

    print("\nText:")
    print(row["text"])

    print("\nSentiment:", result["label"])
    print("Confidence:", round(result["score"], 4))


sentiment_df = pd.DataFrame(sentiment_data)

sentiment_df.to_csv(
    "dataset/sentiment_results.csv",
    index=False
)

print("\nSentiment results saved:")
print("dataset/sentiment_results.csv")


# ==========================================================
# 4. TEXT SUMMARIZATION
# ==========================================================

print("\n" + "=" * 70)
print("2. TEXT SUMMARIZATION")
print("=" * 70)


summarization_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)


summary_data = []

summary_texts = df[df["task"] == "Summarization"]


for _, row in summary_texts.iterrows():

    result = summarization_pipeline(
        row["text"],
        max_length=80,
        min_length=25,
        do_sample=False
    )[0]

    summary = result["summary_text"]

    summary_data.append({
        "id": row["id"],
        "original_text": row["text"],
        "summary": summary
    })

    print("\nOriginal Text:")
    print(row["text"])

    print("\nSummary:")
    print(summary)


summary_df = pd.DataFrame(summary_data)

summary_df.to_csv(
    "dataset/summarized_text.csv",
    index=False
)

print("\nSummarization results saved:")
print("dataset/summarized_text.csv")


# ==========================================================
# 5. ENGLISH TO TAMIL TRANSLATION
# ==========================================================

print("\n" + "=" * 70)
print("3. ENGLISH TO TAMIL TRANSLATION")
print("=" * 70)


# NLLB-200 supports English and Tamil.
# eng_Latn = English
# tam_Taml = Tamil

translation_pipeline = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn",
    tgt_lang="tam_Taml"
)


translation_data = []

translation_texts = df[df["task"] == "Translation"]


for _, row in translation_texts.iterrows():

    result = translation_pipeline(
        row["text"],
        max_length=200
    )[0]

    translated_text = result["translation_text"]

    translation_data.append({
        "id": row["id"],
        "english_text": row["text"],
        "tamil_translation": translated_text
    })

    print("\nEnglish:")
    print(row["text"])

    print("\nTamil:")
    print(translated_text)


translation_df = pd.DataFrame(translation_data)

translation_df.to_csv(
    "dataset/translated_text.csv",
    index=False
)

print("\nTranslation results saved:")
print("dataset/translated_text.csv")


# ==========================================================
# 6. QUESTION ANSWERING
# ==========================================================

print("\n" + "=" * 70)
print("4. QUESTION ANSWERING")
print("=" * 70)


qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


# ----------------------------------------------------------
# Contexts
# ----------------------------------------------------------

context_9 = df.loc[
    df["id"] == 9,
    "text"
].iloc[0]


context_10 = df.loc[
    df["id"] == 10,
    "text"
].iloc[0]


# ----------------------------------------------------------
# Questions
# ----------------------------------------------------------

qa_questions = [

    {
        "id": 9,
        "context": context_9,
        "question": "What library does Hugging Face provide?"
    },

    {
        "id": 9,
        "context": context_9,
        "question": "What tasks does the Transformers library support?"
    },

    {
        "id": 10,
        "context": context_10,
        "question": "What programming language is popular in artificial intelligence?"
    }

]


qa_data = []


for item in qa_questions:

    result = qa_pipeline(
        question=item["question"],
        context=item["context"]
    )

    qa_data.append({
        "id": item["id"],
        "question": item["question"],
        "context": item["context"],
        "answer": result["answer"],
        "confidence": round(result["score"], 4)
    })

    print("\nQuestion:")
    print(item["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nConfidence:")
    print(round(result["score"], 4))


qa_df = pd.DataFrame(qa_data)

qa_df.to_csv(
    "dataset/qa_results.csv",
    index=False
)

print("\nQuestion Answering results saved:")
print("dataset/qa_results.csv")


# ==========================================================
# 7. PIPELINE COMPARISON
# ==========================================================

print("\n" + "=" * 70)
print("5. PIPELINE COMPARISON")
print("=" * 70)


comparison_data = [

    {
        "pipeline": "Sentiment Analysis",
        "input_type": "Text",
        "output_type": "Sentiment label and confidence",
        "application": "Opinion and emotion detection"
    },

    {
        "pipeline": "Text Summarization",
        "input_type": "Long text",
        "output_type": "Short summary",
        "application": "News and document summarization"
    },

    {
        "pipeline": "English to Tamil Translation",
        "input_type": "English text",
        "output_type": "Tamil translation",
        "application": "Language translation"
    },

    {
        "pipeline": "Question Answering",
        "input_type": "Context and question",
        "output_type": "Answer and confidence",
        "application": "Information extraction"
    }

]


comparison_df = pd.DataFrame(comparison_data)


comparison_df.to_csv(
    "dataset/pipeline_comparison.csv",
    index=False
)


print("\nPipeline Comparison:")
print(comparison_df.to_string(index=False))


print("\nPipeline comparison saved:")
print("dataset/pipeline_comparison.csv")


# ==========================================================
# 8. FINAL STATUS
# ==========================================================

print("\n" + "=" * 70)
print("ALL HUGGING FACE PIPELINE TASKS COMPLETED SUCCESSFULLY!")
print("=" * 70)


print("\nGenerated CSV files:")

print("1. dataset/sentiment_results.csv")
print("2. dataset/summarized_text.csv")
print("3. dataset/translated_text.csv")
print("4. dataset/qa_results.csv")
print("5. dataset/pipeline_comparison.csv")


print("\nProject execution completed successfully.")

print("=" * 70)
