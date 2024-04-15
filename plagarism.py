import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Reconstruct text from tokens
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

def calculate_similarity(text1, text2):
    # Preprocess text
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # Vectorize text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity


def check_plagiarism(document1, document2, threshold=0.6):
    # Read contents of documents
    text1 = document1.read()
    text2 = document2.read()

    similarity = calculate_similarity(text1, text2)
    if similarity >= threshold:
        print("Plagiarism detected!\nSimilarity score:", similarity*100,"%")
    else:
        print("No plagiarism detected.\nSimilarity score:", similarity*100,"%")

# Example usage
document1 = open("file1.txt","r")
document2 = open("file2.txt","r")

check_plagiarism(document1, document2)

# Don't forget to close the files after reading
document1.close()
document2.close()
