from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation,MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from string import punctuation
import pandas as pd
import numpy as np
from os import listdir
from collections import Counter
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import os
import tqdm
from tqdm import tqdm

def extra_word(i):
    return (
        i.lower() in set(stopwords.words("english"))
        or i.lower() in punctuation
        or len(i) == 1
        or not (i.isalpha() or i.replace('-','').isalpha()) 
    )

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove punctuation and stopwords
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Join the words back into a single string
    processed_text = ' '.join(words)

    return processed_text

def read_and_preprocess_data(root_folder):
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

    # Create DataFrame
    df = pd.DataFrame(data)

    return df

if __name__ == "__main__":
    print('Started..')

    # Example usage:
    root_folder = 'input'
    preprocessed_data = read_and_preprocess_data(root_folder)

    # Save preprocessed data to CSV
    output_csv = 'preprocessed_data.csv'
    preprocessed_data.to_csv(output_csv, index=False)if __name__ == "__main__":
    print('Started..')
    
    print('Preprocessing done..')
    # Example usage:
    root_folder = 'input'
    preprocessed_data = read_and_preprocess_data(root_folder)

    # Save preprocessed data to CSV
    output_csv = 'preprocessed_data.csv'
    preprocessed_data.to_csv(output_csv, index=False)