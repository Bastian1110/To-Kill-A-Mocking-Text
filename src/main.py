"""
By : Adrian Bravo (Adrian101-hnd)
Project : To-Kill-A-Mocking-Bird
"""

from AbstractSentry import AbstractSentry
from ProAbstractSentry import ProAbstractSentry
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


def validate(sentry : AbstractSentry, dir, tresh):
    # Read a directory full of legit text and plagiarized text, generates a x and y dataset
    x, y = build_validation_dataset(dir)

    # Evaluates the system acurrracy (in a classification problem mode)
    sentry.evaluate_system(x, y, threshold=tresh)

def check_some_docs(sentry, dir, treshhold = 0.75):
    docs = read_directory_files(dir)
    print(f"Analyzing {len(dir)} documents")
    originals = 0 
    total = len(docs)
    x = 0
    for name, text in docs:
        x += 1
        result = sentry.calculate_similarity(text)[1]
        average = sentry.calculate_average_similarity(text)
        originals +=  0 if average > treshhold else 1
        print(f"Name: {name} | Genuine: {'no ' if average > treshhold else 'yes'} | Similar to {result['name']} by {int(result['similarity'] * 100)}% | Average {average}")
    print(f"Result : {originals} of {total} documents are orginal.")
    print(x)


from utils import read_ultimate_dataset


if __name__ == "__main__":
    # Initialize the AbstractSentry object with MongoDB connection details

    #data = read_ultimate_dataset("./datasets/ultimate-dataset", ["text", "name"], {"type" : [6]})
    #document_reals = [[doc["name"], doc["text"]] for doc in data]
    document_reals = read_directory_files("./datasets/uresti_reals")

    model = "sentence-transformers/all-mpnet-base-v2"
    sentry = ProAbstractSentry(no_preprocess,"mongodb://localhost:27017/", "ProSentry" , model, use_sentence_transformer=False)
    sentry.store_embeddings(document_reals)

    check_some_docs(sentry, "./datasets/final-test", treshhold=0.25)
    #validate(sentry, "./datasets/final-test", 0.25)

    #sentry = AbstractSentry(no_preprocess,"mongodb://localhost:27017/", "tfidf")
    #preprocessed_docs = sentry.preprocess_docs(document_reals)
    #sentry.fit_vectorizer(preprocessed_docs)


    #tests = read_directory_files("./datasets/urest")
    #for doc in tests:
    #    res = sentry.calculate_similarity(doc[1])
    #    print(f"Test : {doc[0]} Result : {res[0]['name']} % : {res[0]['similarity']}")

    #check_some_docs(sentry, "./datasets/uresti_reals")

