
# 📘 Relative Answer Accuracy Prediction using Clustering

This project implements an **Answer Accuracy Prediction Model** using **ensemble clustering** and **semantic similarity**. It processes a set of text answers, clusters them using multiple algorithms, and predicts how close a new answer lies to known clusters for a given question — ultimately assigning a relative score or mark.

---

## 🚀 Project Objective

To develop an automated system that can:
- Learn from **previously submitted answers** to questions.
- Group similar answers into **semantic clusters**.
- When a **new answer** is given, predict how closely it matches known high-quality responses.
- Assign a **relative score/mark** to the answer based on:
  - Its **semantic similarity** to cluster centroids.
  - **Consensus confidence** across multiple clustering techniques.

---

## 🔍 Features

- Text Preprocessing: Tokenization, Lemmatization, Stopword removal.
- Feature Extraction: TF-IDF vectorization of cleaned text.
- Clustering:
  - `KMeans`
  - `DBSCAN`
  - `Agglomerative Clustering`
- Ensemble Strategy:
  - Combines clustering results to form a **meta-cluster label**.
- Similarity Matching:
  - Computes cosine similarity with reference/model answers.
- Scoring System:
  - Assigns scores based on semantic similarity and clustering confidence.

---

## 🛠️ Technologies Used

- Python
- `scikit-learn` (Clustering, TF-IDF, Cosine Similarity)
- `NLTK` (Text Processing)
- `pandas`, `numpy`, `tqdm`
- Clustering Ensemble Logic
- File handling via `os`

---

## 📂 Folder Structure

```
project_root/
│
├── answers/                # Root folder with subfolders of categorized answers
│   └── Question1/
│       └── answer1.txt
│       └── answer2.txt
│
├── preprocessed_data.csv   # CSV output of cleaned text answers
├── new.csv                 # TF-IDF vector representation of answers
├── script.py               # Main Python script (your shared code)
└── README.md               # You're here!
```

---

## 📈 How It Works

1. **Data Collection**: Reads text files organized by question folders.
2. **Preprocessing**: Converts each answer to lowercase, lemmatizes, and removes stopwords/punctuation.
3. **TF-IDF Vectorization**: Transforms the text into numerical feature vectors.
4. **Base Clustering**: Performs KMeans, DBSCAN, and Agglomerative Clustering.
5. **Ensemble Meta-Clustering**:
   - Aligns labels across methods.
   - Combines them for robust cluster assignment.
6. **Scoring**:
   - Compares each answer’s embedding with a set of **model answers**.
   - Uses **cosine similarity** and cluster agreement (**confidence**) to calculate a **relative score**.



---

## 📝 Future Improvements

- Dynamic model answer selection per question.
- Integration with web UI for live scoring.
- Use of contextual embeddings (e.g., BERT, SentenceTransformers) for better semantic understanding.
- Incorporation of answer grammar/spelling scoring.

---


