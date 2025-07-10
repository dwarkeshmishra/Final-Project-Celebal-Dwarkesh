#!/usr/bin/env python3
"""
K-Means & LDA Topic Modeling for 20 Newsgroups (Codespaces Structure)
---------------------------------------------------------------------
Author: Your Name
Date: 2025-07-10
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score

# ---------------------- 1. Set Paths ----------------------
script_dir = Path(__file__).parent.resolve()
data_dir = script_dir.parent / "dataset"
extract_dir = data_dir / "20_newsgroups"

if not extract_dir.exists():
    raise FileNotFoundError(f"Expected extracted data at {extract_dir}, but it does not exist.")

# ---------------------- 2. Load Documents ----------------------
print("Loading documents...")
docs = []
labels_true = []

for category in sorted(extract_dir.iterdir()):
    if category.is_dir():
        for file_path in sorted(category.glob("*")):
            if file_path.is_file():
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                        if text:  # Only add non-empty documents
                            docs.append(text)
                            labels_true.append(category.name)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

print(f"Loaded {len(docs)} documents from {len(set(labels_true))} categories.")

if len(docs) == 0:
    raise RuntimeError("No documents loaded. Please check your dataset structure and file contents.")

# ---------------------- 3. Vectorize ----------------------
print("Vectorizing...")
tfidf = TfidfVectorizer(stop_words="english", max_df=0.5, min_df=5)
X_tfidf = tfidf.fit_transform(docs)

count = CountVectorizer(stop_words="english", max_df=0.5, min_df=5)
X_counts = count.fit_transform(docs)

# ---------------------- 4. K-Means Clustering ----------------------
print("Running K-Means...")
k = 20
km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_tfidf)
cluster_ids = km.labels_
sil = silhouette_score(X_tfidf, cluster_ids)
print(f"Silhouette score (k={k}): {sil:.3f}")

# ---------------------- 5. LDA Topic Modeling ----------------------
print("Running LDA...")
n_topics = 20
lda = LatentDirichletAllocation(
    n_components=n_topics,
    learning_method="batch",
    random_state=42)
lda.fit(X_counts)

# Display top words per topic
print("\nTop words per LDA topic:")
words = count.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    top = topic.argsort()[-10:][::-1]
    print(f"Topic {idx:02d}: ", ", ".join(words[i] for i in top))

# ---------------------- 6. Save Outputs ----------------------
print("Saving outputs...")
out_dir = script_dir.parent / "outputs"
out_dir.mkdir(exist_ok=True)

# 6a. Cluster assignments
pd.DataFrame({
    "document": docs,
    "true_label": labels_true,
    "kmeans_cluster": cluster_ids
}).to_csv(out_dir/"kmeans_labels.csv", index=False)

# 6b. Topic proportions
topic_df = pd.DataFrame(
    lda.transform(X_counts),
    columns=[f"topic_{i:02d}" for i in range(n_topics)])
topic_df.to_csv(out_dir/"lda_document_topics.csv", index=False)

print("Done. Results written to 'outputs/' directory.")
