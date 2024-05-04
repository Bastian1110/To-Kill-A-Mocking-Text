"""
By : Harimi Manzano (@HarumiManz) & Sebastian Mora (@Bastian110)
Project : To-Kill-A-Mocking-Bird
"""
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import numpy as np
import torch 
import pickle

class ProAbstractSentry:
    def __init__(self, preprocess, mongo_conn_string, database_name, model, use_sentence_transformer=True):
        self.database = MongoClient(mongo_conn_string)[database_name]
        self.preprocess = preprocess
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.use_sentence_transformer = use_sentence_transformer
        if use_sentence_transformer:
            self.sentence_transformer = SentenceTransformer(model)

    def get_mean_pooling_embeddings(self, text, tokenizer, model):
        tokens = tokenizer.encode_plus(text, max_length=128,
                                        truncation=True, padding='max_length',
                                        return_tensors='pt')
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state
            attention_mask = tokens['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
            mask_embeddings = embeddings * mask
            summed = torch.sum(mask_embeddings, 1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / counts
            return mean_pooled

    def store_embeddings(self, documents):
        docs_texts = [self.preprocess(text) for _, text in documents]
        if self.use_sentence_transformer:
            embeddings = self.sentence_transformer.encode(docs_texts) 
        else :
            embeddings = [self.get_mean_pooling_embeddings(text, self.tokenizer, self.model) for text in docs_texts]
        self.database.vectors.delete_many({})
        for (doc_name, _), embedding in zip(documents, embeddings):
            self.database.vectors.insert_one({"name": doc_name, "data": pickle.dumps(embedding if self.use_sentence_transformer else embedding.cpu().numpy())})


    def calculate_similarity(self, doc_text):
        preprocess_doc = self.preprocess(doc_text)
        if self.use_sentence_transformer:
            input_embedding = self.sentence_transformer.encode(preprocess_doc)
        else :
            input_embedding = self.get_mean_pooling_embeddings(preprocess_doc, self.tokenizer, self.model)
        if input_embedding.ndim == 1:
            input_embedding = input_embedding.reshape(1, -1)
        embeddings, names = self._get_embeddings_from_db()
        similarity_matrix = cosine_similarity(input_embedding, embeddings)[0].tolist()
        sorted_similarity_index = sorted(range(len(similarity_matrix)), key=lambda x: similarity_matrix[x], reverse=True)
        return [{"name": names[i], "similarity": similarity_matrix[i]} for i in sorted_similarity_index]
    
    def calculate_average_similarity(self, doc_text):
        preprocess_doc = self.preprocess(doc_text)
        if self.use_sentence_transformer:
            input_embedding = self.sentence_transformer.encode(preprocess_doc)
        else :
            input_embedding = self.get_mean_pooling_embeddings(preprocess_doc, self.tokenizer, self.model)
        if input_embedding.ndim == 1:
            input_embedding = input_embedding.reshape(1, -1)
        embeddings, names = self._get_embeddings_from_db()
        similarity_matrix = cosine_similarity(input_embedding, embeddings)[0].tolist()
        return sum(similarity_matrix) / len(similarity_matrix)
    
        

    def _get_embeddings_from_db(self):
        embeddings_data = self.database.vectors.find()
        embedding_doc_names = []
        embeddings_list = []
        for emb in embeddings_data:
            embeddings_list.append(pickle.loads(emb["data"]))
            embedding_doc_names.append(emb["name"])
        return np.vstack(embeddings_list), embedding_doc_names

    def evaluate_system(self, x, real_y, graphic=True, threshold=0.25):
        pred_y = []
        for doc in x:
            similarity = self.calculate_average_similarity(doc)
            pred_y.append(0 if similarity < threshold else 1)
        cm = confusion_matrix(real_y, pred_y)
        if graphic:
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap='Purples', xticklabels=['Original', 'Fake'], yticklabels=['Original', 'Fake'])
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()
        return cm
