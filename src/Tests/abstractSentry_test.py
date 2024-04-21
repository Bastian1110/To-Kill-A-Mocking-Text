import unittest
from AbstractSentry import AbstractSentry
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Define the preprocessing function
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

class TestAbstractSentry(unittest.TestCase):
    def setUp(self):
        # Set up the AbstractSentry instance for testing
        self.abstract_sentry = AbstractSentry(stem_and_stopwords, "mongodb://localhost:27017/", "test_database")

    def tearDown(self):
        # Clean up any test data in the database after each test
        self.abstract_sentry.database.vectors.delete_many({})
        self.abstract_sentry.database.vectorizer.delete_many({})

    def test_preprocess_docs_longer_text(self):
        # Test the preprocess_docs method with longer and more nuanced text
        docs = [
            ("doc1", "This is a test document. It contains multiple sentences and some punctuation, such as commas."),
            ("doc2", "Another test document with a longer sentence that includes numbers like 123 and special characters like @#$%.")
        ]
        preprocessed_docs = self.abstract_sentry.preprocess_docs(docs)
        expected_output = [
            ("doc1", "test document contain multipl sentenc punctuat comma"),
            ("doc2", "anoth test document longer sentenc includ number like 123 special charact like")
        ]
        print(preprocessed_docs)
        self.assertEqual(preprocessed_docs, expected_output)
    
    def test_preprocess_docs_empty_text(self):
        # Test the preprocess_docs method with empty text in one document
        docs = [
                ("doc1", ""), 
                ("doc2", "This is a non-empty document.")
                ]
        preprocessed_docs = self.abstract_sentry.preprocess_docs(docs)
        expected_output = [("doc1", ""), ("doc2", "document")]
        self.assertEqual(preprocessed_docs, expected_output)

    def test_preprocess_docs_no_alphanumeric_words(self):
        # Test the preprocess_docs method with documents containing no alphanumeric words after preprocessing
        docs = [
            ("doc1", "This document contains only special characters like @#$%."),
            ("doc2", "1234567890")  # Only numbers
        ]
        preprocessed_docs = self.abstract_sentry.preprocess_docs(docs)
        expected_output = [("doc1", "document contain special charact like"), ("doc2", "1234567890")]
        self.assertEqual(preprocessed_docs, expected_output)


    def test_fit_vectorizer(self):
        # Test the fit_vectorizer method
        docs = [("doc1", "This is a test document."), ("doc2", "Another test document.")]
        vectors_data = list(self.abstract_sentry.database.vectors.find())
        vectors = len(vectors_data)
        self.abstract_sentry.fit_vectorizer(docs)
        vectors_data2 = list(self.abstract_sentry.database.vectors.find())
        # Check if the vectors are stored in the database
        self.assertEqual(len(vectors_data2), vectors + 2)  # Check if two vectors were stored

    def test_calculate_similarity(self):
        # Test the calculate_similarity method
        doc_text = "This is a dummy test text. This a sentence written by someone that has no idea what they are doing. Hey are you still reading this sentence? Hi how are you?"
        similarity_results = self.abstract_sentry.calculate_similarity(doc_text)
        vectors_data = list(self.abstract_sentry.database.vectors.find())
        self.assertEqual(len(similarity_results), len(vectors_data))  # Check if similarity results are returned for all stored documents

    def test_add_new_document(self):
        # Test the add_new_document method
        doc_name = "new_doc"
        doc_text = "This is a new test document. I has two simple sentences."
        vectors_data = list(self.abstract_sentry.database.vectors.find())
        self.abstract_sentry.fit_vectorizer([(doc_name,doc_text)])
        vectors = len(vectors_data)
        vector = self.abstract_sentry.add_new_document(doc_name, doc_text)
        vectors_data2 = list(self.abstract_sentry.database.vectors.find())
        
        self.assertEqual(len(vectors_data2), vectors + 1)  # Check if one vector was added

if __name__ == '__main__':
    unittest.main()