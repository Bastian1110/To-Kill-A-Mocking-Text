import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import numpy as np

class AbstractSentry:
    """
    A class for managing and analyzing text documents with TF-IDF vectorization and cosine similarity.

    Attributes:
    preprocess_function (callable): Function to preprocess document text.
    database (MongoClient.Database): Database client for storing vector data and state.
    vectorizer (TfidfVectorizer): A scikit-learn TF-IDF vectorizer.

    Methods:
    __init__(self, preprocess_function, mongo_conn_string, database_name): Initializes the AbstractSentry instance.
    preprocess_docs(self, docs): Preprocesses a list of documents.
    save_vectorizer_state(self): Saves the state of the TF-IDF vectorizer to the database.
    load_vectorizer_state(self): Loads the TF-IDF vectorizer state from the database.
    fit_vectorizer(self, preprocessed_docs): Fits the vectorizer on preprocessed documents and stores the vectors.
    calculate_similarity(self, doc_text): Calculates and sorts the similarity of a document against stored documents.
    _get_vecs_from_db(self): Retrieves vectors and their associated document names from the database.
    _add_vecs_to_db(self, documents, vectors): Adds document vectors and their names to the database.
    store_vectors(self, documents, vectors): Stores vectors in the database, clearing previous entries.
    add_new_document(self, doc_name, doc_text): Adds a new document to the database after vectorization.
    """

    def __init__(self, preprocess_function, mongo_conn_string, database_name):
        """
        Initializes the AbstractSentry instance with a MongoDB connection and prepares the vectorizer.

        Parameters:
        preprocess_function (callable): Function to preprocess document text.
        mongo_conn_string (str): MongoDB connection string.
        database_name (str): Name of the database to use.
        """
        self.preprocess_function = preprocess_function
        self.database = MongoClient(mongo_conn_string)[database_name]
        self.vectorizer = TfidfVectorizer()
        self.load_vectorizer_state()

    def preprocess_docs(self, docs):
        """
        Applies the preprocessing function to each document in the provided list.

        Parameters:
        docs (list of tuple): A list of tuples, each containing the document name and its corresponding text.

        Returns:
        list of tuple: Preprocessed document names paired with their processed text.
        """
        return [(doc_name, self.preprocess_function(doc_text)) for doc_name, doc_text in docs]

    def save_vectorizer_state(self):
        """
        Saves the current state of the TF-IDF vectorizer to the database.
        """
        vectorizer_state = pickle.dumps(self.vectorizer)
        self.database.vectorizer.update_one({}, {"$set": {"state": vectorizer_state}}, upsert=True)

    def load_vectorizer_state(self):
        """
        Loads the TF-IDF vectorizer state from the database if available; otherwise, initializes a new vectorizer.
        """
        result = self.database.vectorizer.find_one()
        if result:
            self.vectorizer = pickle.loads(result['state'])
        else:
            self.vectorizer = TfidfVectorizer()

    def fit_vectorizer(self, preprocessed_docs):
        """
        Fits the TF-IDF vectorizer with the preprocessed document texts and stores the resulting vectors.

        Parameters:
        preprocessed_docs (list of tuple): A list of preprocessed document names paired with their texts.
        """
        docs_texts = [text for _, text in preprocessed_docs]
        vectors = self.vectorizer.fit_transform(docs_texts)
        self.save_vectorizer_state()
        self.store_vectors(preprocessed_docs, vectors)

    def calculate_similarity(self, doc_text):
        """
        Calculates the cosine similarity of a new document against all stored documents and returns sorted results.

        Parameters:
        doc_text (str): The text of the document to compare.

        Returns:
        list of dict: A list of dictionaries, each containing a document name and its similarity score, sorted by similarity in descending order.
        """
        preprocessed_doc = self.preprocess_function(doc_text)
        input_vector = self.vectorizer.transform([preprocessed_doc])
        vectors, names = self._get_vecs_from_db()
        similarity_matrix = cosine_similarity(input_vector, vectors)[0]
        sorted_similarity_index = sorted(range(len(similarity_matrix)), key=lambda x: similarity_matrix[x], reverse=True)
        return [{"name": names[i], "similarity": similarity_matrix[i]} for i in sorted_similarity_index]

    def _get_vecs_from_db(self):
        """
        Retrieves all document vectors and their names from the database.

        Returns:
        tuple: A tuple containing an array of vectors and a list of document names.
        """
        vectors_data = self.database.vectors.find()
        vector_doc_names = []
        vectors_list = []
        for vec in vectors_data:
            vectors_list.append(pickle.loads(vec["data"]))
            vector_doc_names.append(vec["name"])
        return np.vstack(vectors_list), vector_doc_names

    def _add_vecs_to_db(self, documents, vectors):
        """
        Adds document vectors along with their names to the database.

        Parameters:
        documents (list of tuple): A list of documents and their preprocessed texts.
        vectors (ndarray): An array of document vectors.
        """
        for (doc_name, _), vector in zip(documents, vectors.toarray()):
            self.database.vectors.insert_one({"name": doc_name, "data": pickle.dumps(vector)})

    def store_vectors(self, documents, vectors):
        """
        Stores new document vectors in the database, removing any existing entries first.

        Parameters:
        documents (list of tuple): Documents and their preprocessed texts.
        vectors (ndarray): Document vectors to store.
        """
        self.database.vectors.delete_many({})
        self._add_vecs_to_db(documents, vectors)

    def add_new_document(self, doc_name, doc_text):
        """
        Adds a new document to the database after preprocessing and vectorizing it.

        Parameters:
        doc_name (str): The name of the document.
        doc_text (str): The text of the document.

        Returns:
        ndarray: The vector of the newly added document.
        """
        preprocessed_doc = self.preprocess_function(doc_text)
        vector = self.vectorizer.transform([preprocessed_doc])
        self._add_vecs_to_db([(doc_name, preprocessed_doc)], vector)
        return vector
