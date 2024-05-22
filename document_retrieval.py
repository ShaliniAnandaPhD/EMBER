import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch
from utils import preprocess_text, load_model, log_retrieval_metrics
import logging

#Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetriever:
    def __init__(self, model_path, index_name, es_host, es_port):
        """
        Initialize the DocumentRetriever.

        Args:
            model_path (str): Path to the trained document retrieval model.
            index_name (str): Name of the Elasticsearch index containing the documents.
            es_host (str): Elasticsearch host.
            es_port (int): Elasticsearch port.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = load_model(model_path, AutoModel)
        self.index_name = index_name
        self.es = Elasticsearch(hosts=[{'host': es_host, 'port': es_port}])

    def encode_query(self, query):
        """
        Encode the query using the document retrieval model.

        Args:
            query (str): The query string.

        Returns:
            numpy.ndarray: The encoded query vector.
        """
        try:
            # Preprocess the query
            query = preprocess_text(query)

            # Tokenize the query
            input_ids = self.tokenizer.encode(query, add_special_tokens=True, return_tensors='pt')

            # Generate the query embedding
            with torch.no_grad():
                query_embedding = self.model(input_ids)[0][:, 0, :].numpy()

            return query_embedding

        except Exception as e:
            logging.error(f"Error encoding query: {str(e)}")
            logging.error("Possible solution: Check if the query is properly preprocessed and tokenized.")
            raise e

    def encode_documents(self, documents):
        """
        Encode the documents using the document retrieval model.

        Args:
            documents (list): A list of document strings.

        Returns:
            numpy.ndarray: The encoded document vectors.
        """
        try:
            # Preprocess the documents
            documents = [preprocess_text(doc) for doc in documents]

            # Tokenize the documents
            input_ids = self.tokenizer.batch_encode_plus(documents, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)

            # Generate the document embeddings
            with torch.no_grad():
                document_embeddings = self.model(input_ids['input_ids'])[0][:, 0, :].numpy()

            return document_embeddings

        except Exception as e:
            logging.error(f"Error encoding documents: {str(e)}")
            logging.error("Possible solution: Check if the documents are properly preprocessed and tokenized.")
            raise e

    def retrieve_documents(self, query, top_k=5):
        """
        Retrieve the top-k most relevant documents for a given query.

        Args:
            query (str): The query string.
            top_k (int): The number of documents to retrieve.

        Returns:
            list: A list of retrieved document IDs.
        """
        try:
            # Encode the query
            query_embedding = self.encode_query(query)

            # Search for relevant documents using Elasticsearch
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
                            "params": {"query_vector": query_embedding.tolist()}
                        }
                    }
                }
            }
            search_results = self.es.search(index=self.index_name, body=search_body, size=top_k)

            # Extract the retrieved document IDs
            retrieved_docs = [hit['_id'] for hit in search_results['hits']['hits']]

            # Log retrieval metrics
            log_retrieval_metrics(query, retrieved_docs)

            return retrieved_docs

        except Exception as e:
            logging.error(f"Error retrieving documents: {str(e)}")
            logging.error("Possible solution: Check if Elasticsearch is running and the index exists.")
            raise e

    def update_embeddings(self):
        """
        Update the document embeddings in the Elasticsearch index.
        """
        try:
            # Retrieve all documents from the index
            query = {"query": {"match_all": {}}}
            documents = self.es.search(index=self.index_name, body=query, size=10000)

            # Extract the document texts
            docs = [hit['_source']['text'] for hit in documents['hits']['hits']]

            # Encode the documents
            document_embeddings = self.encode_documents(docs)

            # Update the embeddings in the index
            for hit, embedding in zip(documents['hits']['hits'], document_embeddings):
                doc_id = hit['_id']
                self.es.update(index=self.index_name, id=doc_id, body={"doc": {"embedding": embedding.tolist()}})

            logging.info("Document embeddings updated successfully.")

        except Exception as e:
            logging.error(f"Error updating document embeddings: {str(e)}")
            logging.error("Possible solution: Check if Elasticsearch is running and the index exists.")
            raise e

# Main entry point
if __name__ == '__main__':
    model_path = 'path/to/trained/document_retrieval_model'
    index_name = 'wildfire_documents'
    es_host = 'localhost'
    es_port = 9200

    retriever = DocumentRetriever(model_path, index_name, es_host, es_port)

    # Example usage
    query = "What are the evacuation procedures for wildfire emergencies?"
    retrieved_docs = retriever.retrieve_documents(query, top_k=5)
    print(f"Retrieved documents for query: {query}")
    print(retrieved_docs)

    # Update document embeddings
    retriever.update_embeddings()
