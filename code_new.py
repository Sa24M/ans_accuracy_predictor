import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import nltk
from tqdm import tqdm

# Define function to get WordNet POS tags based on nltk POS tags
def get_word_pos(tag):
    if tag.startswith('j'):
        return wordnet.ADJ
    elif tag.startswith('v'):
        return wordnet.VERB
    elif tag.startswith('n'):
        return wordnet.NOUN
    elif tag.startswith('r'):
        return wordnet.ADV
    else:
        return None

# Stopwords set
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    words = word_tokenize(text)

    # Lemmatization with selected POS tags
    lemmatizer = WordNetLemmatizer()
    words_lemmatized = []

    # Iterate over the words
    for word in words:
        # Get the part-of-speech tag for the word
        word_pos = nltk.pos_tag([word])[0][1][0].lower()

        # Map the POS tag to WordNet POS tag
        w_pos = get_word_pos(word_pos)

        # Lemmatize the word if a valid POS tag is found
        if w_pos is not None:
            words_lemmatized.append(lemmatizer.lemmatize(word, w_pos))
        else:
            # If no valid POS tag is found, add the word as it is
            words_lemmatized.append(word)

    # Remove punctuation and stopwords
    words_filtered = [word for word in words_lemmatized if word.isalnum() and word not in stop_words]

    # Join the words back into a single string
    processed_text = ' '.join(words_filtered)

    return processed_text

# Function to read and preprocess data
def read_and_preprocess_data(root_folder, output_csv):
    data = {'Folder': [], 'File': [], 'Text': []}

    for folder_name in tqdm(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        preprocessed_text = preprocess_text(text)

                        # Store data in dictionary
                        data['Folder'].append(folder_name)
                        data['File'].append(file_name)
                        data['Text'].append(preprocessed_text)

    # Create DataFrame and save preprocessed data to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    return df

# Main function to execute the entire workflow
if __name__ == "__main__":
    print('Started..')

    # Example root folder and output CSV
    root_folder_path = 'answers'
    output_csv_file = 'preprocessed_data.csv'
    output_tfidf_csv_file = 'new.csv'

    # Preprocess data and save to CSV
    df = read_and_preprocess_data(root_folder_path, output_csv_file)

    # Create TfidfVectorizer and calculate TF-IDF values
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Text'])

    # Convert TF-IDF matrix to a DataFrame for better readability
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Save TF-IDF values to CSV
    tfidf_df.to_csv(output_tfidf_csv_file, index=False)

    # Clustering Section: Apply KMeans, DBSCAN, Agglomerative Clustering

    X = tfidf_df.values  # Use the TF-IDF matrix as input embeddings

    # Normalize if not already done
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply base clustering algorithms
    kmeans_labels = KMeans(n_clusters=3, random_state=0).fit_predict(X_scaled)
    dbscan_labels = DBSCAN(eps=1.5, min_samples=2).fit_predict(X_scaled)
    agglo_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X_scaled)

    # Ensemble label matrix
    ensemble_matrix = np.vstack((kmeans_labels, dbscan_labels, agglo_labels)).T
    noise_fix_label = np.max(ensemble_matrix) + 1
    ensemble_matrix[ensemble_matrix == -1] = noise_fix_label

    # Meta clustering
    meta_cluster = AgglomerativeClustering(n_clusters=3)
    final_labels = meta_cluster.fit_predict(ensemble_matrix)

    # Confidence score computation
    def compute_confidence(row):
        values, counts = np.unique(row, return_counts=True)
        return counts.max() / len(row)

    confidence_scores = np.apply_along_axis(compute_confidence, 1, ensemble_matrix)

    # Model answer vectors (example: pick 3 good samples from your set)
    model_answer_vectors = X_scaled[:3]  # Replace with real model answers if available

    # Compute final centroids for each cluster
    def compute_centroids(X, labels):
        centroids = []
        for label in np.unique(labels):
            centroids.append(np.mean(X[labels == label], axis=0))
        return np.array(centroids)

    final_centroids = compute_centroids(X_scaled, final_labels)

    # Match clusters to model answers
    sim_matrix = cosine_similarity(final_centroids, model_answer_vectors)
    final_to_model = np.argmax(sim_matrix, axis=1)

    # Scoring
    marks = []
    full_marks = 5
    partial_marks = 2.5

    for i in range(len(X_scaled)):
        predicted_cluster = final_labels[i]
        matched_model_index = final_to_model[predicted_cluster]
        sim = cosine_similarity([X_scaled[i]], [model_answer_vectors[matched_model_index]])[0][0]

        if sim > 0.9:
            mark = full_marks * confidence_scores[i]
        elif sim > 0.7:
            mark = partial_marks * confidence_scores[i]
        else:
            mark = 0.5 * confidence_scores[i]

        marks.append(round(mark, 2))

    # Output results to a DataFrame
    results_df = pd.DataFrame({
        'Answer': df['Text'],  # Assuming the 'Text' column is the answers
        'Final_Cluster': final_labels,
        'Confidence': np.round(confidence_scores, 2),
        'Marks': marks
    })

    print(results_df.to_string(index=False))
