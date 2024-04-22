"""
By : Adrian Bravo (Adrian101-hnd)
Project : To-Kill-A-Mocking-Bird
"""

from AbstractSentry import AbstractSentry
from utils import read_directory_files, build_validation_dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize a PorterStemmer and a set of English stopwords for text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stem_and_stopwords(doc: str) -> str:
    """
    Applies stemming and removes stopwords from the given document text.

    Parameters:
    doc (str): The original document text.

    Returns:
    str: A preprocessed version of the document where each word is stemmed and stopwords are removed.
    """
    words = nltk.word_tokenize(doc.lower())
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()]
    return ' '.join(filtered_words)

def no_preprocess(doc :str) -> str:
    return doc


def fit_system(sentry):
    # Read and preprocess real documents from a directory
    documents_reals = read_directory_files("./datasets/reals")
    documents_fakes = read_directory_files("./datasets/fakes")

    # Preprocess the real documents using the specified preprocessing function
    preprocessed_docs = sentry.preprocess_docs(documents_reals)

    # Fit the vectorizer on the preprocessed real documents and store the vectors in the database
    sentry.fit_vectorizer(preprocessed_docs)

    # Take a sample document from the fake documents dataset for testing
    test = documents_reals[0]
    print("Testing document:", test[0])  # Print the name of the document being tested

    # Calculate the similarity of the test document against the stored real document vectors
    res = sentry.calculate_similarity(test[1])

    # Print the similarity results
    for doc in res:
        print(doc)


def validate(sentry : AbstractSentry):
    # Read a directory full of legit text and plagiarized text, generates a x and y dataset
    x, y = build_validation_dataset("./datasets/valid")

    # Evaluates the system acurrracy (in a classification problem mode)
    sentry.evaluate_system(x, y, threshold=0.25)

if __name__ == "__main__":
    # Initialize the AbstractSentry object with MongoDB connection details
    sentry = AbstractSentry(no_preprocess, "mongodb://localhost:27017/", "TKAMT")

    fit_system(sentry)
    validate(sentry)

